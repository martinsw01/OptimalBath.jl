export solve

using SinFVM, ElasticArrays

function zero_loss(β)
    return zero(eltype(β))
end

struct SweOptimizationProblem
    make_backend
    initial_bathymetry
    grid
    reconstruction
    timestepper
    interior_objective
    terminal_objective
    parameter_objective
    u0
    T
    function SweOptimizationProblem(N,
                                    u0,
                                    T;
                                    interior_objective = NoObjective(),
                                    terminal_objective = NoObjective(),
                                    parameter_objective = zero_loss,
                                    make_backend=SinFVM.make_cpu_backend,
                                    initial_bathymetry=zeros(N + 1),
                                    grid=SinFVM.CartesianGrid(N; gc=2, boundary=SinFVM.WallBC()),
                                    reconstruction=SinFVM.LinearReconstruction(1.),
                                    timestepper=SinFVM.RungeKutta2())
        @assert length(initial_bathymetry) == N + 1 "Bathymetry must have length N+1=$(N + 1) ≠ $(length(initial_bathymetry))"
        return new(make_backend,
                   initial_bathymetry,
                   grid,
                   reconstruction,
                   timestepper,
                   interior_objective,
                   terminal_objective,
                   parameter_objective,
                   u0,
                   T)
    end       
end

function _create_simulator(problem, β)
    FloatType = eltype(β)
    backend = problem.make_backend(FloatType)
    b = vcat(zeros(FloatType, 2),
             problem.initial_bathymetry .+ β,
             zeros(FloatType, 2))
    bathymetry = SinFVM.BottomTopography1D(b, backend, problem.grid)
    equation = SinFVM.ShallowWaterEquations1D(bathymetry)
    numericalflux = SinFVM.CentralUpwind(equation)
    conserved_system = SinFVM.ConservedSystem(backend, problem.reconstruction, numericalflux, equation, problem.grid, [SinFVM.SourceTermBottom()])
    simulator = SinFVM.Simulator(backend, conserved_system, problem.timestepper, problem.grid, cfl=0.2)
    return simulator
end

function solve_primal(problem::SweOptimizationProblem, β)
    FloatType = eltype(β)
    simulator = _create_simulator(problem, β)

    x = SinFVM.cell_centers(problem.grid)
    initial = [FloatType.(U) for U in problem.u0.(x)]
    SinFVM.set_current_state!(simulator, initial)

    U = ElasticMatrix(reshape(SinFVM.current_interior_state(simulator), :, 1))
    t = ElasticVector([zero(FloatType)])

    function collect_state(t_n, simulator)
        append!(U, SinFVM.current_interior_state(simulator))
        append!(t, t_n)
    end

    SinFVM.simulate_to_time(simulator, problem.T; callback=collect_state)
    
    return U, t, SinFVM.cell_faces(problem.grid)
end