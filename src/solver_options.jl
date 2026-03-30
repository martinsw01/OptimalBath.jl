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

function assert_correct_sizes(N, U0, initial_bathymetry)
    @assert length(initial_bathymetry) == N + 1 "Bathymetry must have length N+1=$(N + 1) ≠ $(length(initial_bathymetry))"
    @assert size(U0.U, 1) == N "Initial states must have size N=$(N) in the first dimension, but got $(size(U0.U, 1))"
end

"""
    PrimalSWEProblem(N, U0, T; initial_bathymetry=zeros(N + 1), domain=[0.0 1.0])
A data structure representing the primal shallow water equations parameters.
"""
struct PrimalSWEProblem{Bathymetry, FloatType, InitialStates<:AverageElevationStates, Domain}
    initial_bathymetry::Bathymetry
    T::FloatType
    U0::InitialStates
    domain::Domain
    N::Int64
    function PrimalSWEProblem(N, U0, T; initial_bathymetry=zeros(N + 1), domain=[0.0 1.0])
        assert_correct_sizes(N, U0, initial_bathymetry)
        return new{typeof(initial_bathymetry), typeof(T), typeof(U0), typeof(domain)}(initial_bathymetry, T, U0, domain, N)
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