export SoftL1

"""
    SoftL1(α)

A smooth approximation to the L1 regularization, divided by the number of elements in `β`.
`α` controls the sharpness of the approximation.
"""
struct SoftL1 <: Regularization
    α::Float64
end

function (r::SoftL1)(β)
    return sum(soft_abs_fn(r.α), β) / length(β)
end

# log(1+exp(x))
function log1pexp(x)
    alpha = max(1, x)
    return alpha + log(exp(-alpha) + exp(x-alpha))
end


soft_abs(α, x) = (log1pexp(-α * x) + log1pexp(α * x) - log(4)) / α
soft_abs_fn(α) = x -> soft_abs(α, x)

soft_abs_derivative(α, x) = tanh(0.5 * α * x)

function add_gradient!(g, β, scale, r::SoftL1)
    g .+= scale .* soft_abs_derivative.(r.α, β) ./ length(β)
end