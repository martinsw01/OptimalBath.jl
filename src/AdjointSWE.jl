export solve_adjoint

using StaticArrays

const desingularizing_kappa = 1e-5

function desingularize(h, p)
    h_star = desingularize(h)
    return p / h_star
end

function desingularize(h)
    h_star = copysign(1, h)*max(abs(h), min(h^2/(2*desingularizing_kappa) + desingularizing_kappa/2.0, desingularizing_kappa))
    return h_star
end

function swe_jacobian_transpose(h, hu)
    u = desingularize(h, hu)
    return @SMatrix [0.0 9.81*h - u^2; 1.0 2*u]
end

function flux(Λ, U)
    dfᵀ = swe_jacobian_transpose(U...)
    return -dfᵀ * Λ
end

function compute_eigenvalues(h, hu)
    u = desingularize(h, hu)
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

const AverageDepthStates{T, N, A} = States{Average, Depth, T, N, A}

function solve_adjoint(Λ0, U::AverageDepthStates, dJdU, b, t, Δx)
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

const LeftDepthStates{T, N, A} = States{Left, Depth, T, N, A}
const RightDepthStates{T, N, A} = States{Right, Depth, T, N, A}

wet(Ul⁺, Uc⁻, Uc⁺, Ur⁻) = !left_interface_dry(Ul⁺, Uc⁻) && !right_interface_dry(Uc⁺, Ur⁻)
right_interface_dry(Uc⁺, Ur⁻) = height(Uc⁺) < desingularizing_kappa || height(Ur⁻) < desingularizing_kappa
left_interface_dry(Ul⁺, Uc⁻) = height(Ul⁺) < desingularizing_kappa || height(Uc⁻) < desingularizing_kappa
only_left_dry(Ul⁺, Uc⁻, Uc⁺, Ur⁻) = left_interface_dry(Ul⁺, Uc⁻) && !right_interface_dry(Uc⁺, Ur⁻)
only_right_dry(Ul⁺, Uc⁻, Uc⁺, Ur⁻) = right_interface_dry(Uc⁺, Ur⁻) && !left_interface_dry(Ul⁺, Uc⁻)


function solve_adjoint(Λ0, Ul::LeftDepthStates, Ur::RightDepthStates, dJdU, b, t, Δx)
    Ul = Ul.U
    Ur = Ur.U
    Λ = similar(Ul)

    N, M = size(Ul)
    Λ[:, end] .= Λ0
    for n in M:-1:2
        Δt = t[n] - t[n-1]
        Λ[1, n-1] = compute_next_Λ_left_boundary(Λ[1:2, n]...,
                                                 Ul[1:2, n]...,
                                                 Ur[1, n],
                                                 dJdU[1, n],
                                                 b[1:2]...,
                                                 Δt, Δx)
        for i in 2:N-1
            stencil = Ur[i-1, n], Ul[i, n], Ur[i, n], Ul[i+1, n]
            if wet(stencil...)
                 Λ[i, n-1] = compute_next_Λ(Λ[i-1:i+1, n]...,
                                           Ul[i:i+1, n]...,
                                           Ur[i-1:i, n]...,
                                           dJdU[i, n],
                                           b[i:i+1]...,
                                           Δt, Δx)
            elseif only_left_dry(stencil...)
                Λ[i, n-1] = compute_next_Λ_left_boundary(Λ[i:i+1, n]...,
                                                        Ul[i:i+1, n]...,
                                                        Ur[i, n],
                                                        dJdU[i, n],
                                                        b[i:i+1]...,
                                                        Δt, Δx)
            elseif only_right_dry(stencil...)
                Λ[i, n-1] = compute_next_Λ_right_boundary(Λ[i-1:i, n]...,
                                                        Ul[i, n],
                                                        Ur[i-1:i, n]...,
                                                        dJdU[i, n],
                                                        b[i:i+1]...,
                                                        Δt, Δx)
            else
                Λ[i, n-1] = Λ[i, n]
            end
        end
        Λ[N, n-1] = compute_next_Λ_right_boundary(Λ[N-1:N, n]...,
                                                  Ul[N, n],
                                                  Ur[N-1:N, n]...,
                                                  dJdU[N, n],
                                                  b[N:N+1]...,
                                                  Δt, Δx)
    end
    return Λ
end

function compute_next_Λ(Λl, Λc, Λr, Uc⁻, Ur⁻, Ul⁺, Uc⁺, dJdU, bl, br, Δt, Δx)
    return Λc + Δt * compute_semi_discrete_derivative(Λl, Λc, Λr, Uc⁻, Ur⁻, Ul⁺, Uc⁺, dJdU, bl, br, Δt, Δx)
end


function compute_next_Λ_left_boundary(Λc, Λr, Uc⁻, Ur⁻, Uc⁺, dJdU, bl, br, Δt, Δx)
    Λl = compute_adjoint_ghost_cell(Λc, Λr, Uc⁻)
    # Λl = compute_ghost_cell(Λc, nothing) #compute_adjoint_ghost_cell(Λc, Λr, Uc⁻)
    Ul⁺ = compute_ghost_cell(Uc⁻, nothing)
    return compute_next_Λ(Λl, Λc, Λr, Uc⁻, Ur⁻, Ul⁺, Uc⁺, dJdU, bl, br, Δt, Δx)
end

function compute_next_Λ_right_boundary(Λl, Λc, Uc⁻, Ul⁺, Uc⁺, dJdU, bl, br, Δt, Δx)
    Λr = compute_adjoint_ghost_cell(Λc, Λl, Uc⁺)
    # Λr = compute_ghost_cell(Λc, nothing) #compute_ghost_cell(Λc, nothing)
    Ur⁻ = compute_ghost_cell(Uc⁺, nothing)
    return compute_next_Λ(Λl, Λc, Λr, Uc⁻, Ur⁻, Ul⁺, Uc⁺, dJdU, bl, br, Δt, Δx)
end

function compute_semi_discrete_derivative(Λl, Λc, Λr, Uc⁻, Ur⁻, Ul⁺, Uc⁺, dJdU, bl, br, Δt, Δx)        
    Fl = numerical_flux(Λl, Λc, Ul⁺, Uc⁻)
    Fr = numerical_flux(Λc, Λr, Uc⁺, Ur⁻)
    dfₓᵀΛ = compute_flux_source_improved(Λc, Uc⁻, Uc⁺, Δx)
    SᵀΛ = compute_bathymetry_source_improved(Λc, Uc⁻, Uc⁺, bl, br, Δx)
    return - (Fr - Fl)/Δx - dfₓᵀΛ + SᵀΛ + dJdU
end

function compute_S2(Ul, Ur, bl, br, Δx)
    hl = height(Ul)
    hr = height(Ur)

    if hl < desingularizing_kappa && hr > desingularizing_kappa
        return S2 = 9.81 * hr / Δx
    elseif hr < desingularizing_kappa && hl > desingularizing_kappa
        return S2 = - 9.81 * hl / Δx
    else
        return S2 = - 9.81 * (br - bl) / Δx
    end
end

function compute_bathymetry_source_improved(Λc, Ul, Ur, bl, br, Δx)
    λ2 = Λc[2]
    S2 = compute_S2(Ul, Ur, bl, br, Δx)
    return State(S2 * λ2, 0.0)
end

function compute_flux_source_improved(Λ, Ul, Ur, Δx)
    hl, pl = Ul
    hr, pr = Ur

    # Regularization:
    if hl < Δx
        pl *= hl
    end
    if hr < Δx
        pr *= hr
    end
        
    dfₓᵀ = (swe_jacobian_transpose(hr, pr) - swe_jacobian_transpose(hl, pl)) / Δx
    return dfₓᵀ * Λ
end