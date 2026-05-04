export ForwardADGradient, ForwardDiffBackend

import ForwardDiff

ForwardDiffBackend() = DI.AutoForwardDiff()
const ForwardADGradient{Preparation} = ADGradient{Preparation, <:DI.AutoForwardDiff}
ForwardADGradient(args...) = ADGradient(ForwardDiffBackend(), args...)

# Ensure the same element is selected when there is no unique maximum
function Base.maximum(x::AbstractArray{<:ForwardDiff.Dual})
    i = findmax(ForwardDiff.value, x)[2]
    return x[i]
end