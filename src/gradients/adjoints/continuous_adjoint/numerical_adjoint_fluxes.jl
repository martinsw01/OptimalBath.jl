function add_adjoint_flux!(Λ, U, n, Δt, dir, grid, da::ContinuousAdjointSWE)
    Δx = get_Δx(grid, dir)
    for_each_left_boundary_directional_stencil(dir, grid) do center, right
        Λ[center, n] += compute_left_flux_contribution(Λ[center, n+1], Λ[right, n+1],
                                                       U[center, n], U[right, n],
                                                       Δt, Δx, dir, da)
    end
    for_each_interior_directional_stencil(dir, grid) do left, center, right
        Λ[center, n] += compute_flux_contribution(Λ[left, n+1], Λ[center, n+1], Λ[right, n+1],
                                                  U[left, n], U[center, n], U[right, n],
                                                  Δt, Δx, dir, da)
    end
    for_each_right_boundary_directional_stencil(dir, grid) do left, center
        Λ[center, n] += compute_right_flux_contribution(Λ[left, n+1], Λ[center, n+1],
                                                        U[left, n], U[center, n],
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
# function compute_eigenvalues(U, dir, da::ContinuousAdjointSWE)
#     primal_eq = primal_equation(da)
#     return VolumeFluxes.compute_eigenvalues(primal_eq, dir, U...)
# end

function numerical_adjoint_flux(Λl, Λr, Ul, Ur, Δt, Δx, dir, da::ContinuousAdjointSWE)
    # TODO: Handle dry states
    fl = adjoint_flux(Λl, Ul, dir, da)
    fr = adjoint_flux(Λr, Ur, dir, da)

    eig_left⁺, eig_left⁻ = compute_eigenvalues(Ul, dir, da)
    eig_right⁺, eig_right⁻ = compute_eigenvalues(Ur, dir, da)
    a⁺ = max(eig_left⁺, eig_right⁺, 0.0)
    a⁻ = min(eig_left⁻, eig_right⁻, 0.0)

    return (a⁺ * fl - a⁻ * fr + a⁻ * a⁺ * (Λr - Λl)) / (a⁺ - a⁻)
end

function compute_flux_contribution(Λl, Λc, Λr, Ul, Uc, Ur, Δt, Δx, dir, da::ContinuousAdjointSWE)
    Fr = numerical_adjoint_flux(Λc, Λr, Uc, Ur, Δt, Δx, dir, da)
    Fl = numerical_adjoint_flux(Λl, Λc, Ul, Uc, Δt, Δx, dir, da)
    return (Fr - Fl) * Δt / Δx
end

function compute_left_flux_contribution(Λc, Λr, Uc, Ur, Δt, Δx, dir, da::ContinuousAdjointSWE)
    # TODO: adjoint ghost cells
    Λl_ghost = compute_ghost_cell(Λc, dir)
    Ul_ghost = compute_ghost_cell(Uc, dir)
    return compute_flux_contribution(Λl_ghost, Λc, Λr, Ul_ghost, Uc, Ur, Δt, Δx, dir, da)
end

function compute_right_flux_contribution(Λl, Λc, Ul, Uc, Δt, Δx, dir, da::ContinuousAdjointSWE)
    # TODO: adjoint ghost cells
    Λr_ghost = compute_ghost_cell(Λc, dir)
    Ur_ghost = compute_ghost_cell(Uc, dir)
    return compute_flux_contribution(Λl, Λc, Λr_ghost, Ul, Uc, Ur_ghost, Δt, Δx, dir, da)
end