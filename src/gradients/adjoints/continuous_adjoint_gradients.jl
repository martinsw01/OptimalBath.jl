export AdjointApproachGradient, compute_gradient!

include("continuous_adjoint_swe.jl")

struct ContinuousAdjointGradient{BathymetryBuffer} <: AdjointGradient
    bathymetry_buffer::BathymetryBuffer
    function ContinuousAdjointGradient(bathymetry)
        buffer = copy(bathymetry)
        return new{typeof(buffer)}(buffer)
    end
end

function adjoint_solver(::PrimalSWESolver, ::ContinuousAdjointGradient)
    return ContinuousAdjointSWE()
end

function compute_gradient!(G, Λ, U, t, Δx, objectives::Objectives, ::ContinuousAdjointGradient)
    compute_gradient!(G, Λ, U, t, objectives.design_indices)
end

function compute_gradient!(G, Λ, Ul, Ur, t, Δx, objectives::Objectives, ::ContinuousAdjointGradient)
    compute_gradient!(G, Λ, Ul, Ur, t, objectives.design_indices)
end


function integrate(U, Λ, t, j)
    g = 9.81
    return sum(height.(U[j,2:end]) .* momentum.(Λ[j, 2:end]) .* diff(t)) * g
end

function _compute_gradient!(G, Λ, U, t, design_indices::Colon)
    G .= zero(eltype(G))
    N, M = size(U)
    for j in 1:N
        temp_integral = integrate(U, Λ, t, j)
        G[j] -= temp_integral
        G[j+1] += temp_integral
    end
end

function _compute_gradient!(G, Λ, U, t, design_indices)
    N, M = size(U)
    for (i, j) in enumerate(design_indices)
        if j == 1
            G[i] = -integrate(U, Λ, t, j)
        elseif j == N + 1
            G[i] = integrate(U, Λ, t, j - 1)
        else
            G[i] = integrate(U, Λ, t, j - 1) - integrate(U, Λ, t, j)
        end
    end
end

function compute_gradient!(G, Λ, U::States{Average, Depth}, t, design_indices)
    _compute_gradient!(G, Λ, U.U, t, design_indices)
end

function compute_gradient!(G, Λ, Ul::States{Left, Depth}, Ur::States{Right, Depth}, t, design_indices)
    _compute_gradient!(G, Λ, 0.5 .* (Ul.U .+ Ur.U), t, design_indices)
end