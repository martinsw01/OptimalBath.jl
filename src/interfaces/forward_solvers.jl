export PrimalSWESolver, SolverBackend, solve_primal
export Reconstruction, NoReconstruction, LinearReconstruction
export TimeStepper, ForwardEuler, RK2
export BathymetrySourceTerm, DefaultBathymetrySource

abstract type SolverBackend end

abstract type Reconstruction end

abstract type TimeStepper end

abstract type BathymetrySourceTerm end

struct DefaultBathymetrySource <: BathymetrySourceTerm end

abstract type PrimalSWESolver{R<:Reconstruction, TS<:TimeStepper, BS<:BathymetrySourceTerm} end

struct LinearReconstruction <: Reconstruction end

struct NoReconstruction <: Reconstruction end

struct ForwardEuler <: TimeStepper end

struct RK2 <: TimeStepper end

"""
    solve_primal(solver::PrimalSWESolver, δb[, callback])

Solves the primal shallow water equations with the provided `solver` given the change `δb` in bathymetry, returning the solution `U`, time points `t`, and spatial points `x`.
"""
function solve_primal end

"""
    compute_Δx(solver::PrimalSWESolver)
Computes the spatial discretization step size `Δx` for the given `solver`.
"""
function compute_Δx end

"""
    create_callback(f, solver::PrimalSWESolver)
Creates a callback function for the given `solver` that applies the function `f(U_n::States, t_n, Δt)` at each time step during the primal solve.
"""
function create_callback end

"""
    initial_state(solver::PrimalSWESolver)
"""
function initial_state end

"""
    get_bathymetry(solver::PrimalSWESolver)
"""
function get_bathymetry end