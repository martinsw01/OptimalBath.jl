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

    @warn "Uses only one side of the reconstruction of U"
    U = States{Average, Depth}(Ul.U)
    compute_gradient!(G, Λ, U, t, Δx, objectives.design_indices)
    objective = compute_objective(U, t, x, β, objectives)
end

function update_bathymetry!(aag::AdjointApproachGradient, indices, β)
    aag.bathymetry_buffer[indices] .+= β
end

function _height(U)
    return U[1]
end

function _adjoint_momentum(Λ)
    return Λ[2]
end

function compute_gradient!(G, Λ, U::States{Average, Depth, T, D, A}, t, Δx, design_indices::Colon) where {T, D, A}
    U = U.U
    G .= zero(eltype(G))
    N, M = size(U)
    for j in 1:N
        temp_integral = integrate(U, Λ, t, j)
        G[j] -= temp_integral
        G[j+1] += temp_integral
    end
end


function integrate(U, Λ, t, j)
    g = 9.81
    return sum(_height.(U[j,2:end]) .* _adjoint_momentum.(Λ[j, 2:end]) .* diff(t)) * g
end

function compute_gradient!(G, Λ, U::States{Average, Depth, T, D, A}, t, Δx, design_indices) where {T, D, A}
    U = U.U
    N, M = size(U)
    g = 9.81
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