export solve_adjoint

using StaticArrays

function swe_jacobian_transpose(h, hu)
    return @SMatrix [0.0 9.81*h - (hu/h)^2; 1.0 2*(hu/h)]
end

function flux(Λ, U)
    dfᵀ = swe_jacobian_transpose(U...)
    return -dfᵀ * Λ
end

function compute_eigenvalues(h, hu)
    u = hu / h
    c = sqrt(9.81 * h)
    return u + c, u - c
end

function numerical_flux(Λl, Λr, Ul, Ur)
    fl = flux(Λl, Ul)
    fr = flux(Λr, Ur)

    eig_left = compute_eigenvalues(Ul...)
    eig_right = compute_eigenvalues(Ur...)

    a⁺ = max(0, eig_left[1], eig_right[1])
    a⁻ = min(0, eig_left[2], eig_right[2])

    return (a⁺ * fl - a⁻ * fr + a⁻ * a⁺ * (Λr - Λl)) / (a⁺ - a⁻)
end


function compute_adjoint_ghost_cell(Λ_interior, Λ, U_interior)
    h = U_interior[1]
    if h > 0
        return compute_ghost_cell(Λ_interior, Λ)
    else
        return copy(Λ_interior)
    end
end

function compute_ghost_cell(U_interior, U)
    return @SVector [U_interior[1], -U_interior[2]]
end

"""
    compute_flux_source(Λ, Ul, Ur, Δx)

Computes the source due to the reformulation of the adjoint pde to conservative form.
"""
function compute_flux_source(Λ, Ul, Ur, Δx)
    dfₓᵀ = (swe_jacobian_transpose(Ur...) - swe_jacobian_transpose(Ul...)) / (2 * Δx)
    return dfₓᵀ * Λ
end

function compute_bathymetry_source(Λ, bl, br, Δx)
    λ2 = Λ[2]
    S2 = - 9.81 * (br - bl) / Δx
    return @SVector [S2 * λ2, 0.0]
end

function compute_next_Λ_left_boundary(Λc, Λr, Uc, Ur, dJdU, bl, br, Δt, Δx)
    Λl = compute_adjoint_ghost_cell(Λc, Λr, Uc)
    Ul = compute_ghost_cell(Uc, Ur)
    return compute_next_Λ(Λl, Λc, Λr, Ul, Uc, Ur, dJdU, bl, br, Δt, Δx)
end

function compute_next_Λ_right_boundary(Λl, Λc, Ul, Uc, dJdU, bl, br, Δt, Δx)
    Λr = compute_adjoint_ghost_cell(Λc, Λl, Uc)
    Ur = compute_ghost_cell(Uc, Ul)
    return compute_next_Λ(Λl, Λc, Λr, Ul, Uc, Ur, dJdU, bl, br, Δt, Δx)
end

function compute_semi_discrete_derivative(Λl, Λc, Λr, Ul, Uc, Ur, dJdU, bl, br, Δt, Δx)
    Fl = numerical_flux(Λl, Λc, Ul, Uc)
    Fr = numerical_flux(Λc, Λr, Uc, Ur)
    dfₓᵀΛ = compute_flux_source(Λc, Ul, Ur, Δx)
    SᵀΛ = compute_bathymetry_source(Λc, bl, br, Δx)
    return - (Fr - Fl)/Δx - dfₓᵀΛ + SᵀΛ + dJdU
end

function compute_next_Λ(Λl, Λc, Λr, Ul, Uc, Ur, dJdU, bl, br, Δt, Δx)
    Λc + Δt * compute_semi_discrete_derivative(Λl, Λc, Λr, Ul, Uc, Ur, dJdU, bl, br, Δt, Δx)
end


function solve_adjoint(Λ0, U::States{Average, Depth, T, D, A}, dJdU, b, t, Δx) where {T, D, A}
    U = U.U
    Λ = similar(U)
    N, M = size(U)
    Λ[:, end] .= Λ0
    for n in M:-1:2
        Δt = t[n] - t[n-1]
        Λ[1, n-1] = compute_next_Λ_left_boundary(Λ[1:2, n]...,
                                                 U[1:2, n]...,
                                                 dJdU[1, n],
                                                 b[1:2]...,
                                                 Δt, Δx)
        for i in 2:N-1
            Λ[i, n-1] = compute_next_Λ(Λ[i-1:i+1, n]...,
                                       U[i-1:i+1, n]...,
                                       dJdU[i, n],
                                       0.25*(b[i-1] + b[i]),
                                       0.25*(b[i+1] + b[i+2]),
                                       Δt, Δx)
        end
        Λ[N, n-1] = compute_next_Λ_right_boundary(Λ[N-1:N, n]...,
                                                  U[N-1:N, n]...,
                                                  dJdU[N, n],
                                                  b[N:N+1]...,
                                                  Δt, Δx)
    end
    return Λ
end