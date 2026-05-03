export MooncakeGradient

using Mooncake
using DifferentiationInterface

struct MooncakeGradient{GradientBuffer} <: ADGradient end

function gradient!(f, g, β, spec, objectives, ::MooncakeGradient)
    backend = AutoMooncake(; config=nothing)
    objective, _ = value_and_gradient!(f, g, backend, β, Const(spec), Const(objectives))
    return objective
end