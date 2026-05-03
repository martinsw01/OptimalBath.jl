export ForwardADGradient

using ForwardDiff, DiffResults, PreallocationTools

"""
    ForwardADGradient(bathymetry::AbstractArray, β::AbstractArray)

A gradient computation strategy using ForwardDiff for automatic differentiation.
Stores two preallocated buffers: one for the bathymetry and one for the gradient results.
The bathymetry starts as a copy of the initial bathymetry, and is updated during gradient computations.
"""
struct ForwardADGradient{GradientBuffer, BathymetryBuffer} <: ADGradient
    gradient_buffer::GradientBuffer
    bathymetry_buffer::BathymetryBuffer
    function ForwardADGradient(bathymetry::AbstractArray, β::AbstractArray=bathymetry)
        gradient_buffer = create_gradient_buffer(β)
        bathymetry_buffer = create_bathymetry_buffer(bathymetry, β)
        GB = typeof(gradient_buffer)
        BB = typeof(bathymetry_buffer)
        return new{GB, BB}(gradient_buffer, bathymetry_buffer)
    end
end

function create_gradient_buffer(β)
    value = first(β)
    derivs = similar(β)
    gradient_buffer = DiffResults.DiffResult(value, derivs)
    return gradient_buffer
end

function create_bathymetry_buffer(bathymetry, β)
    return PreallocationTools.DiffCache(bathymetry)
end

function gradient!(f, g, β, spec, objectives, ad::ForwardADGradient)
    ForwardDiff.gradient!(ad.gradient_buffer, β -> f(β, spec, objectives), β)
    g .= ad.gradient_buffer.derivs[1]
    return ad.gradient_buffer.value
end

function update_and_get_bathymetry!(ad::ForwardADGradient, swe_problem::PrimalSWEProblem, indices, β)
    bathymetry = get_tmp(ad.bathymetry_buffer, first(β))
    bathymetry[indices] .= β + swe_problem.initial_bathymetry[indices]
    return bathymetry
end

# Ensure the same element is selected when there is no unique maximum
function Base.maximum(x::AbstractArray{<:ForwardDiff.Dual})
    i = findmax(ForwardDiff.value, x)[2]
    return x[i]
end