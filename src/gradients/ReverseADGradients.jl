using ReverseDiff, DiffResults

struct ReverseADGradient{GradientBuffer} <: ADGradient
    gradient_buffer::GradientBuffer
    function ReverseADGradient(β)
        value = first(β)
        derivs = similar(β)
        gradient_buffer = DiffResults.DiffResult(value, derivs)
        GB = typeof(gradient_buffer)
        return new{GB}(gradient_buffer)
    end
end

function update_and_get_bathymetry!(ad::ReverseADGradient, swe_problem::PrimalSWEProblem, indices, β)
    bathymetry = eltype(β).(swe_problem.initial_bathymetry)
    bathymetry[indices] .+= β
    return bathymetry
end

function gradient!(ad::ReverseADGradient, f, β)
    ReverseDiff.gradient!(ad.gradient_buffer, f, β)
end

function get_gradient(ad::ReverseADGradient)
    return ad.gradient_buffer.derivs[1]
end

function get_objective(ad::ReverseADGradient)
    return ad.gradient_buffer.value
end
