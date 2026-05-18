function add_flux_spatial_derivative!(Λ, U, n, Δt, dir, grid, da::ContinuousAdjointSWE)
    # TODO: Handle dry states
    # - Perhaps no nothing when state is dry (height(U) < depth_cutoff(da))?
    Δx = get_Δx(grid, dir)
    for_each_left_boundary_directional_stencil(dir, grid) do center, right
        Λ[center, n] += compute_left_flux_jacobian_contribution(Λ[center, n+1],
                                                                U[center, n], U[right, n],
                                                                Δt, Δx, dir, da)
    end
    for_each_interior_directional_stencil(dir, grid) do left, center, right
        Λ[center, n] += compute_flux_jacobian_contribution(Λ[center, n+1],
                                                           U[left, n], U[center, n], U[right, n],
                                                           Δt, Δx, dir, da)
    end
    for_each_right_boundary_directional_stencil(dir, grid) do left, center
        Λ[center, n] += compute_right_flux_jacobian_contribution(Λ[center, n+1],
                                                                 U[left, n], U[center, n],
                                                                 Δt, Δx, dir, da)
    end
end

function compute_flux_jacobian_contribution(Λc, Ul, Uc, Ur, Δt, Δx, dir, da::ContinuousAdjointSWE)
    return flux_jacobian_spatial_derivative(Ul, Ur, Δx, dir, da) * Λc .* Δt
end

function compute_left_flux_jacobian_contribution(Λc, Uc, Ur, Δt, Δx, dir, da::ContinuousAdjointSWE)
    Ul_ghost = compute_ghost_cell(Uc, dir)
    return compute_flux_jacobian_contribution(Λc, Ul_ghost, Uc, Ur, Δt, Δx, dir, da)
end

function compute_right_flux_jacobian_contribution(Λc, Ul, Uc, Δt, Δx, dir, da::ContinuousAdjointSWE)
    Ur_ghost = compute_ghost_cell(Uc, dir)
    return compute_flux_jacobian_contribution(Λc, Ul, Uc, Ur_ghost, Δt, Δx, dir, da)
end

function flux_jacobian_spatial_derivative(Ul, Ur, Δx, dir, da::ContinuousAdjointSWE)
    dfdUᵀ_l = primal_flux_jacobian_transpose(Ul, dir, da)
    dfdUᵀ_r = primal_flux_jacobian_transpose(Ur, dir, da)
    return (dfdUᵀ_r .- dfdUᵀ_l) ./ (2Δx)
end