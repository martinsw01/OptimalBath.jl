function add_flux_spatial_derivative!(Λ, U_left, U_right, n, Δt, dir, grid, da::ContinuousAdjointSWE)
    # TODO: Handle dry states
    # - Perhaps do nothing when state is dry (height(U) < depth_cutoff(da))?
    Δx = get_Δx(grid, dir)
    for_each_left_boundary_directional_stencil(dir, grid) do center, right
        Λ[center, n] += compute_left_flux_jacobian_contribution(Λ[center, n+1],
                                                                U_left[center], U_right[center], U_left[right],
                                                                Δt, Δx, dir, da)
    end
    for_each_interior_directional_stencil(dir, grid) do left, center, right
        Λ[center, n] += compute_flux_jacobian_contribution(Λ[center, n+1],
                                                           U_right[left], U_left[center], U_right[center], U_left[right],
                                                           Δt, Δx, dir, da)
    end
    for_each_right_boundary_directional_stencil(dir, grid) do left, center
        Λ[center, n] += compute_right_flux_jacobian_contribution(Λ[center, n+1],
                                                                 U_right[left], U_left[center], U_right[center],
                                                                 Δt, Δx, dir, da)
    end
end

function compute_flux_jacobian_contribution(Λc, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da::ContinuousAdjointSWE)
    return flux_jacobian_spatial_derivative(U_right_left, U_left_center, U_right_center, U_left_right, Δx, dir, da.flux_jacobian_divergence, da) * Λc .* Δt
end

function compute_left_flux_jacobian_contribution(Λc, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da::ContinuousAdjointSWE)
    U_right_left = compute_ghost_cell(U_left_center, dir)
    return compute_flux_jacobian_contribution(Λc, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da)
end

function compute_right_flux_jacobian_contribution(Λc, U_right_left, U_left_center, U_right_center, Δt, Δx, dir, da::ContinuousAdjointSWE)
    U_left_right = compute_ghost_cell(U_right_center, dir)
    return compute_flux_jacobian_contribution(Λc, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da)
end

"""
    flux_jacobian_spatial_derivative(U_right_left, U_left_center, U_right_center, U_left_right, Δx, dir, ::FluxJacobianDivergence, ::ContinuousAdjointSWE)

Approximates the derivative of the flux Jacobian along `dir`.

## Arguments
- `U_right_left`: Right reconstruction of the left state
- `U_left_center`: Left reconstruction of the center state
- `U_right_center`: Right reconstruction of the center state
- `U_left_right`: Left reconstruction of the right state
- `Δx`: Grid spacing in the direction of interest
- `dir`: Direction of differentiation
- `::FluxJacobianDivergence`: Type of flux Jacobian divergence to use (for dispatch)
- `::ContinuousAdjointSWE`: Type indicator for dispatch
"""
function flux_jacobian_spatial_derivative end

include("balanced_flux_jacobian_divergence.jl")
include("central_difference_flux_jacobian.jl")