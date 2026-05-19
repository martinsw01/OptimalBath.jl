export AdjointApproachGradient, compute_gradient!

include("continuous_adjoint_swe.jl")

function ContinuousAdjointGradient(primal_solver::PrimalSWESolver)
    return AdjointApproachGradient(ContinuousAdjointSWE(primal_solver))
end