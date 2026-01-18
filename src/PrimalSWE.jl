using SinFVM, ElasticArrays

"""
    SinFVMPrimalSWEProblem

Wrapper around SinFVM. Contains problem definition and solver parameters.
"""
struct SinFVMPrimalSWEProblem <: PrimalSWEProblem
    _make_backend
    initial_bathymetry
    _grid
    _reconstruction
    _timestepper
    u0
    T
    function SinFVMPrimalSWEProblem(N,
                              u0,
                              T;
                              domain = [0.0 1.0],
                              _make_backend=SinFVM.make_cpu_backend,
                              initial_bathymetry=zeros(N + 1),
                              _grid=SinFVM.CartesianGrid(N; gc=2, boundary=SinFVM.WallBC(), extent = domain),
                              _reconstruction=SinFVM.LinearReconstruction(1.),
                              _timestepper=SinFVM.RungeKutta2())
        @assert length(initial_bathymetry) == N + 1 "Bathymetry must have length N+1=$(N + 1) ≠ $(length(initial_bathymetry))"
        return new(_make_backend,
                   initial_bathymetry,
                   _grid,
                   _reconstruction,
                   _timestepper,
                   u0,
                   T)
    end       
end

function _create_simulator(problem, β)
    FloatType = eltype(β)
    backend = problem._make_backend(FloatType)
    b = vcat(zeros(FloatType, 2),
             problem.initial_bathymetry .+ β,
             zeros(FloatType, 2))
    bathymetry = SinFVM.BottomTopography1D(b, backend, problem._grid)
    equation = SinFVM.ShallowWaterEquations1D(bathymetry)
    numericalflux = SinFVM.CentralUpwind(equation)
    conserved_system = SinFVM.ConservedSystem(backend, problem._reconstruction, numericalflux, equation, problem._grid, [SinFVM.SourceTermBottom()])
    simulator = SinFVM.Simulator(backend, conserved_system, problem._timestepper, problem._grid, cfl=0.2)
    return simulator
end

function solve_primal(problem::SinFVMPrimalSWEProblem, β = zero(problem.initial_bathymetry))
    FloatType = eltype(β)
    simulator = _create_simulator(problem, β)

    SinFVM.set_current_state!(simulator, problem.u0)

    U = ElasticMatrix(reshape(SinFVM.current_interior_state(simulator), :, 1))
    t = ElasticVector([zero(FloatType)])

    function collect_state(t_n, simulator)
        append!(U, SinFVM.current_interior_state(simulator))
        append!(t, t_n)
    end

    SinFVM.simulate_to_time(simulator, problem.T; callback=collect_state)
    
    return U, t, SinFVM.cell_faces(problem._grid)
end