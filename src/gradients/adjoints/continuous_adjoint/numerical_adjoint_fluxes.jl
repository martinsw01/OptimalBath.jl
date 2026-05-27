function add_adjoint_flux!(Λ, U_left, U_right, n, Δt, dir, grid, da::ContinuousAdjointSWE)
    Δx = get_Δx(grid, dir)
    for_each_left_boundary_directional_stencil(dir, grid) do center, right
        Λ[center, n] += compute_left_flux_contribution(Λ[center, n+1], Λ[right, n+1],
                                                       U_left[center], U_right[center], U_left[right],
                                                       Δt, Δx, dir, da)
    end
    for_each_interior_directional_stencil(dir, grid) do left, center, right
        Λ[center, n] += compute_flux_contribution(Λ[left, n+1], Λ[center, n+1], Λ[right, n+1],
                                                  U_right[left], U_left[center], U_right[center], U_left[right],
                                                  Δt, Δx, dir, da)
    end
    for_each_right_boundary_directional_stencil(dir, grid) do left, center
        Λ[center, n] += compute_right_flux_contribution(Λ[left, n+1], Λ[center, n+1],
                                                        U_right[left], U_left[center], U_right[center],
                                                        Δt, Δx, dir, da)
    end
end

function primal_flux_jacobian_transpose(U::State{2}, ::XDIRT, da::ContinuousAdjointSWE)
    h, hu = U
    u = desingularize(h, hu, da)
    return @SMatrix [0.0  9.81h - u^2;
                     1.0  2u]
end

function adjoint_flux(Λ, U, dir, da::ContinuousAdjointSWE)
    dfdUᵀ = primal_flux_jacobian_transpose(U, dir, da)
    return dfdUᵀ * Λ
end

function compute_eigenvalues(U::State{2}, ::XDIRT, da::ContinuousAdjointSWE)
    h, hu = U
    u = desingularize(h, hu, da)
    c = sqrt(9.81 * h)
    return u + c, u - c
end

function compute_eigenvalues(U::State{3}, dir::Val, da::ContinuousAdjointSWE)
    h = height(U)
    hu = momentum(U, dir)
    u = desingularize(h, hu, da)
    c = sqrt(9.81 * h)
    return u + c, u - c, u
end

function numerical_adjoint_flux(Λl, Λr, Ul, Ur, Δt, Δx, dir, da::ContinuousAdjointSWE)
    # TODO: Handle dry states
    fl = adjoint_flux(Λl, Ul, dir, da)
    fr = adjoint_flux(Λr, Ur, dir, da)

    eig_left⁺, eig_left⁻ = compute_eigenvalues(Ul, dir, da)
    eig_right⁺, eig_right⁻ = compute_eigenvalues(Ur, dir, da)
    a⁺ = max(eig_left⁺, eig_right⁺, 0.0)
    a⁻ = min(eig_left⁻, eig_right⁻, 0.0)

    if a⁺ - a⁻ < 1e-6
        return zero(Λl)
    end

    return (a⁺ * fl - a⁻ * fr + a⁻ * a⁺ * (Λr - Λl)) / (a⁺ - a⁻)
end

function compute_flux_contribution(Λl, Λc, Λr, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da::ContinuousAdjointSWE)
    Fr = numerical_adjoint_flux(Λc, Λr, U_right_center, U_left_right , Δt, Δx, dir, da)
    Fl = numerical_adjoint_flux(Λl, Λc, U_right_left, U_left_center, Δt, Δx, dir, da)
    return (Fr - Fl) * Δt / Δx
end

function compute_left_flux_contribution(Λc, Λr, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da::ContinuousAdjointSWE)
    # Λl_ghost = compute_adjoint_ghost_cell(Λc, Λr, U_left_center, dir, da)
    Λl_ghost = compute_adjoint_ghost_cell(Λc, U_left_center, dir, da)
    U_right_left = compute_ghost_cell(U_left_center, dir)
    return compute_flux_contribution(Λl_ghost, Λc, Λr, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da)
end

function compute_right_flux_contribution(Λl, Λc, U_right_left, U_left_center, U_right_center, Δt, Δx, dir, da::ContinuousAdjointSWE)
    # Λr_ghost = compute_adjoint_ghost_cell(Λc, Λl, U_right_center, dir, da)
    Λr_ghost = compute_adjoint_ghost_cell(Λc, U_right_center, dir, da)
    U_left_right = compute_ghost_cell(U_right_center, dir)
    return compute_flux_contribution(Λl, Λc, Λr_ghost, U_right_left, U_left_center, U_right_center, U_left_right, Δt, Δx, dir, da)
end

function compute_adjoint_ghost_cell(Λ, U, dir, da::ContinuousAdjointSWE)
    if height(U) < depth_cutoff(da)
        return Λ
    else
        return compute_ghost_cell(Λ, dir)
    end
end

function compute_adjoint_ghost_cell(Λl, Λr, U, dir, da::ContinuousAdjointSWE)
    if height(U) < depth_cutoff(da)
        return 2Λl - Λr
    else
        return compute_ghost_cell(Λl, dir)
    end
end