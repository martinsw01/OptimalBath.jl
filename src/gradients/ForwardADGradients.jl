using ForwardDiff, DiffResults, PreallocationTools

"""
    ForwardADGradient(bathymetry::AbstractArray, β::AbstractArray)

A gradient computation strategy using ForwardDiff for automatic differentiation.
Stores two preallocated buffers: one for the bathymetry and one for the gradient results.
The bathymetry starts as a copy of the initial bathymetry, and is updated during gradient computations.
"""
struct ForwardADGradient{GradientBuffer, BathymetryBuffer} <: GradientType
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


function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem, objectives::Objectives, ad::ForwardADGradient)
    function solve_and_compute_objective(β)
        # # update_bathymetry!(ad, objectives.design_indices, β)
        # bathymetry = zeros(eltype(β), size(ad.bathymetry_buffer))
        # bathymetry .= ad.bathymetry_buffer
        # # bathymetry = ad.bathymetry_buffer
        # # bathymetry[objectives.design_indices] .= β
        bathymetry = update_and_get_bathymetry!(ad, objectives.design_indices, β)

        U, t, x = solve_primal(primal_swe_problem, bathymetry)
        objective = compute_objective(U, t, x, β, objectives)
        return objective
    end
    ForwardDiff.gradient!(ad.gradient_buffer, solve_and_compute_objective, β)
    objective = get_objective(ad)
    G .= get_gradient(ad)
    return objective
end

function update_and_get_bathymetry!(ad::ForwardADGradient, indices, β)
    bathymetry = get_tmp(ad.bathymetry_buffer, first(β))
    bathymetry[indices] .= β
    return bathymetry
end

function get_gradient(ad::ForwardADGradient)
    return ad.gradient_buffer.derivs[1]
end

function get_objective(ad::ForwardADGradient)
    return ad.gradient_buffer.value
end
