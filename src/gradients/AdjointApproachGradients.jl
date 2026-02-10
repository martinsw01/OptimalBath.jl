export AdjointApproachGradient, compute_gradient!

struct AdjointApproachGradient{BathymetryBuffer} <: GradientType 
    bathymetry_buffer::BathymetryBuffer
    function AdjointApproachGradient(bathymetry)
        buffer = copy(bathymetry)
        return new{typeof(buffer)}(buffer)
    end
end

function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem, objectives::Objectives, aag::AdjointApproachGradient)
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, length(initial_state(primal_swe_problem).U))

    (Ul, Ur), t, x = solve_primal(primal_swe_problem, δb)

    dJdU = zero(Ul.U)
    dJdU[objectives.objective_indices, :] .+= 0.5 * objective_density_gradient(objectives.interior_objective, Ul, objectives.objective_indices, :)
    dJdU[objectives.objective_indices, :] .+= 0.5 * objective_density_gradient(objectives.interior_objective, Ur, objectives.objective_indices, :)
    
    Δx = x[2] - x[1]
    Λ0 = zero(Ul.U[:, end])
    Λ0[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ul, objectives.objective_indices, lastindex(t))
    Λ0[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ur, objectives.objective_indices, lastindex(t))

    Λ = solve_adjoint(Λ0, Ul, Ur, dJdU, primal_swe_problem.initial_bathymetry .+ δb, t, Δx)

    compute_gradient!(G, Λ, Ul, Ur, t, objectives.design_indices)
    objective = compute_objective(Ul, Ur, t, x, β, objectives)
end

function update_bathymetry!(aag::AdjointApproachGradient, indices, β)
    aag.bathymetry_buffer[indices] .+= β
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

function compute_gradient!(G, Λ, U::States{Average, Depth, T, D, A}, t, design_indices) where {T, D, A}
    _compute_gradient!(G, Λ, U.U, t, design_indices)
end

function compute_gradient!(G, Λ, Ul::States{Left, Depth, T, D, A}, Ur::States{Right, Depth, T, D, A}, t, design_indices) where {T, D, A}
    _compute_gradient!(G, Λ, 0.5 .* (Ul.U .+ Ur.U), t, design_indices)
end