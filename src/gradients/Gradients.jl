abstract type GradientType end

function compute_objective_and_gradient!(G, β, ::SweOptimizationProblem, ::GradientType) end

include("AdjointApproachGradients.jl")
include("ForwardADGradients.jl")