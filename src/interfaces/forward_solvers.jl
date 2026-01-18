export PrimalSWEProblem, solve_primal


abstract type PrimalSWEProblem end

"""
    solve_primal(primal_swe_problem::PrimalSWEProblem, bathymetry)

Solves the primal shallow water equations for a given `primal_swe_problem` and `bathymetry`, returning the solution `U`, time points `t`, and spatial points `x`.
"""
function solve_primal end