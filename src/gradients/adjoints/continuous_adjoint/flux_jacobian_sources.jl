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
    return flux_jacobian_spatial_derivative(U_right_left, U_left_center, U_right_center, U_left_right, Δx, dir, da) * Λc .* Δt
end

function compute_left_flux_jacobian_contribution(Λc, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da::ContinuousAdjointSWE)
    U_right_left = compute_ghost_cell(U_left_center, dir)
    return compute_flux_jacobian_contribution(Λc, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da)
end

function compute_right_flux_jacobian_contribution(Λc, U_right_left, U_left_center, U_right_center, Δt, Δx, dir, da::ContinuousAdjointSWE)
    U_left_right = compute_ghost_cell(U_right_center, dir)
    return compute_flux_jacobian_contribution(Λc, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da)
end

function flux_jacobian_spatial_derivative(U_right_left, U_left_center, U_right_center, U_left_right, Δx, dir, da::ContinuousAdjointSWE)
    dfdUᵀ_r = interface_jacobian(U_right_center, U_left_right, dir, da)
    dfdUᵀ_l = interface_jacobian(U_right_left, U_left_center, dir, da)
    return -(dfdUᵀ_r .- dfdUᵀ_l) ./ Δx
end


function interface_jacobian(Ul, Ur, dir, da::ContinuousAdjointSWE)
    fl = primal_flux_jacobian_transpose(Ul, dir, da)
    fr = primal_flux_jacobian_transpose(Ur, dir, da)

    eig_left⁺, eig_left⁻ = compute_eigenvalues(Ul, dir, da)
    eig_right⁺, eig_right⁻ = compute_eigenvalues(Ur, dir, da)

    a⁺ = max(eig_left⁺, eig_right⁺, 0.0)
    a⁻ = min(eig_left⁻, eig_right⁻, 0.0)

    if a⁺ - a⁻ < 1e-6
        return zero(fl)
    end

    return (a⁺ * fl - a⁻ * fr) / (a⁺ - a⁻)
end