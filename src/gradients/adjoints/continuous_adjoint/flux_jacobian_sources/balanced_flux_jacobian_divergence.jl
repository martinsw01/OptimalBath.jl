function flux_jacobian_spatial_derivative(U_right_left, U_left_center, U_right_center, U_left_right, Δx, dir, ::BalancedFluxJacobianDivergence, da::ContinuousAdjointSWE)
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