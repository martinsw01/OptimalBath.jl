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

function recording_callback(problem::SinFVMPrimalSWEProblem,
                            U0::States{Average, Elevation, T}, t0) where T
    U = States{Average, Elevation}(ElasticMatrix(reshape(U0.U, :, 1)))
    t = ElasticVector([t0])

    record_state = create_callback(problem) do U_n, t_n, Δt
        append!(U.U, U_n.U)
        append!(t, t_n)
    end

    return record_state, U, t
end

function create_callback(f, ::SinFVMPrimalSWEProblem)
    function callback(t_n, simulator)
        U = SinFVM.current_interior_state(simulator)
        Δt = SinFVM.current_timestep(simulator)
        f(States{Average, Elevation}(U), t_n, Δt)
    end
    return callback
end

function solve_primal(problem::SinFVMPrimalSWEProblem, β)
    FloatType = eltype(β)
    t0 = zero(FloatType)
    callback, U, t = recording_callback(problem, problem.u0, t0)
    solve_primal(problem, β, callback)
    x = CellFaces(SinFVM.cell_faces(problem._grid))
    return U, t, x
end

function solve_primal(problem::SinFVMPrimalSWEProblem, β, callback)
    simulator = _create_simulator(problem, β)

    SinFVM.set_current_state!(simulator, problem.u0)

    SinFVM.simulate_to_time(simulator, problem.T; callback=callback)
end

function compute_Δx(problem::SinFVMPrimalSWEProblem)
    return SinFVM.compute_dx(problem._grid)
end

function initial_state(problem::SinFVMPrimalSWEProblem)
    return States{Average, Elevation}(problem.u0)
end