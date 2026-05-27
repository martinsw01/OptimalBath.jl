export InnerCentralDifferenceFlux, OuterCentralDifferenceFlux

"""
    InnerCentralDifferenceFlux <: FluxJacobianDivergence

Approximates the flux Jacobian divergence `∇ · df/dU^T` using the left and right reconstruction of the center state.

!!! warning "InnerCentralDifferenceFlux is not balanced"
    This flux Jacobian divergence is not balanced against the CentralUpwind numerical flux.
"""
struct InnerCentralDifferenceFlux <: FluxJacobianDivergence end

function flux_jacobian_spatial_derivative(U_right_left, U_left_center, U_right_center, U_left_right, Δx, dir, ::InnerCentralDifferenceFlux, da::ContinuousAdjointSWE)
    dfdUᵀ_r = primal_flux_jacobian_transpose(U_right_center, dir, da)
    dfdUᵀ_l = primal_flux_jacobian_transpose(U_left_center, dir, da)
    return -(dfdUᵀ_r .- dfdUᵀ_l) ./ Δx
end

"""
    OuterCentralDifferenceFlux <: FluxJacobianDivergence

Approximates the flux Jacobian divergence `∇ · df/dU^T` using the left and right average states. Used with `NoReconstruction`.

!!! warning "OuterCentralDifferenceFlux is not balanced"
    This flux Jacobian divergence is not balanced against the CentralUpwind numerical flux.
"""
struct OuterCentralDifferenceFlux <: FluxJacobianDivergence end

function flux_jacobian_spatial_derivative(U_right_left, U_left_center, U_right_center, U_left_right, Δx, dir, ::OuterCentralDifferenceFlux, da::ContinuousAdjointSWE)
    dfdUᵀ_r = primal_flux_jacobian_transpose(U_left_right, dir, da)
    dfdUᵀ_l = primal_flux_jacobian_transpose(U_right_left, dir, da)
    return -(dfdUᵀ_r .- dfdUᵀ_l) ./ (2Δx)
end