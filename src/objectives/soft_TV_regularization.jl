export SoftTV

struct SoftTV <: Regularization
    α::Float64
end

function (r::SoftTV)(β::AbstractVector)
    acc = zero(eltype(β))
    @inbounds for i in firstindex(β):lastindex(β)-1
        acc += soft_abs(r.α, β[i+1] - β[i])
    end
    return acc
end

function add_gradient!(g, β, scale, r::SoftTV)
    @inbounds for i in firstindex(β):lastindex(β)-1
        d = soft_abs_derivative(r.α, β[i+1] - β[i])
        g[i] -= scale * d
        g[i+1] += scale * d
    end
end