module DiscreteAdjoints

using ..OptimalBath: AdjointApproachGradient

include("discrete_adjoint_swe.jl")

export DiscreteAdjointGradient

function DiscreteAdjointGradient(primal_solver::VolumeFluxesSolver)
    return AdjointApproachGradient(DiscreteAdjointSWE(primal_solver))
end

end