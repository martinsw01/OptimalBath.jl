export SolverOptions, PrimalSWEProblem, SolverSpec, build_solver

struct SolverOptions{R<:Reconstruction, TS<:TimeStepper, BS<:BathymetrySourceTerm}
    reconstruction::R
    timestepper::TS
    bathymetry_source::BS
    function SolverOptions(reconstruction::Reconstruction=NoReconstruction(),
                           timestepper::TimeStepper=ForwardEuler(),
                           bathymetry_source::BathymetrySourceTerm=DefaultBathymetrySource())
        return new{typeof(reconstruction), typeof(timestepper), typeof(bathymetry_source)}(reconstruction, timestepper, bathymetry_source)
    end
end

function assert_correct_dimensions(U0, b, grid)
    @assert size(b) == grid.N .+ 1 "Bathymetry must have length $(grid.N .+ 1) but got $(length(b))"
    @assert size(U0.U) == grid.N "Initial states must have size $(grid.N) but got $(size(U0.U))"
end

"""
    PrimalSWEProblem(U0, T, grid; initial_bathymetry=zeros(grid.N))
    PrimalSWEProblem(N, U0, T; initial_bathymetry=zeros(N + 1), domain=[0.0 1.0])
A data structure representing the primal shallow water equations parameters.
"""
struct PrimalSWEProblem{Bathymetry, FloatType, InitialStates<:AverageElevationStates, GridType}
    initial_bathymetry::Bathymetry
    T::FloatType
    U0::InitialStates
    grid::GridType
    function PrimalSWEProblem(U0, T, grid::Grid, initial_bathymetry=zeros(grid.N))
        assert_correct_dimensions(U0, initial_bathymetry, grid)
        return new{typeof(initial_bathymetry), typeof(T), typeof(U0), typeof(grid)}(initial_bathymetry, T, U0, grid)
    end
    function PrimalSWEProblem(N, U0, T; initial_bathymetry=zeros(N + 1), domain=[0.0 1.0])
        grid = Grid1D(N; domain=domain)
        return PrimalSWEProblem(U0, T, grid, initial_bathymetry)
    end
end

struct SolverSpec{SWEProblem<:PrimalSWEProblem, Backend<:SolverBackend, SO<:SolverOptions}
    problem::SWEProblem
    backend::Backend
    solver_options::SO
    function SolverSpec(problem::PrimalSWEProblem, backend::SolverBackend, solver_options::SolverOptions=SolverOptions())
        return new{typeof(problem), typeof(backend), typeof(solver_options)}(problem, backend, solver_options)
    end
end

"""
    build_solver(spec::SolverSpec[, float_type::Type=Float64])
"""
function build_solver end

build_solver(spec::SolverSpec) = build_solver(spec, Float64)