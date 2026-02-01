export PrimalSWEProblem, solve_primal


abstract type PrimalSWEProblem end

"""
    solve_primal(primal_swe_problem::PrimalSWEProblem, bathymetry, callbacks=[])

Solves the primal shallow water equations for a given `primal_swe_problem` and `bathymetry`, returning the solution `U`, time points `t`, and spatial points `x`.
"""
function solve_primal end

"""
    compute_Δx(primal_swe_problem::PrimalSWEProblem)
Computes the spatial discretization step size `Δx` for the given `primal_swe_problem`.
"""
function compute_Δx end

"""
    create_callback(f, primal_swe_problem::PrimalSWEProblem)
Creates a callback function for the given `primal_swe_problem` that applies the function `f(U_n::States, t_n, Δt)` at each time step during the primal solve.
"""
function create_callback end

"""
    initial_state(primal_swe_problem::PrimalSWEProblem)
"""
function initial_state end