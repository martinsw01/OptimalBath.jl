function flatten(U::AbstractMatrix)
    return reduce(vcat, vec(U))
end

function flatten(U::AbstractVector)
    return reduce(vcat, U)
end

function unflatten(u_flat)
    N = length(u_flat) ÷ 2
    [State(u_flat[2i-1:2i]) for i in 1:N]
end

function unflatten(u_flat, N)
    M = length(u_flat) ÷ (2*N)
    reshape(unflatten(u_flat), N, M)
end

using ForwardDiff

function tangent_linear_model(N, U0, δU0, bathymetry)
    function f(U)
        β = zeros(eltype(eltype(U)), N+1)
        U = States{Average, Elevation}(unflatten(U))
        problem = SinFVMPrimalSWEProblem(N, U, 0.1, reconstruction=NoReconstruction(), timestepper=ForwardEuler(); initial_bathymetry=bathymetry)
        U, t, x = solve_primal(problem, β)
        return U, t, x, problem
    end
    function g(U)
        return flatten(f(U)[1].U)
    end

    J_flat = ForwardDiff.jacobian(g, flatten(U0))

    U, t, x, problem = f(flatten(U0))

    δU = unflatten(J_flat * flatten(δU0), N)
    return U, δU, t, problem
end

using LinearAlgebra: dot
function adjoint_dot_test(N, bathymetry)
    U0 = rand(State{Float64}, N)
    U0 = States{Average, Elevation}(U0)
    δU0 = rand(State{Float64}, N)
    U, δU, t, problem = tangent_linear_model(N, U0.U, δU0, bathymetry)

    U = unsafe_to_depth!(U, bathymetry)
    β = zeros(N+1)

    objectives = Objectives()
    Λ0 = rand(State{Float64}, N)
    Δx = 1/N

    Λ = solve_adjoint(Λ0, U, objectives, bathymetry, t, Δx, DiscreteAdjointSWE(problem))

    adjoint_dot_product_test = dot(δU[:, end], Λ0)
    @test adjoint_dot_product_test ≈ dot(δU0, Λ[:, 1])
end

function compute_perturbation_with_nonzero_objective(Λ, U, δU, Δx, t, objectives)
    U = U.U
    N, M = size(U)
    Λ_temp = similar(U, N)

    J_final = OptimalBath.DiscreteAdjoints.compute_objective_step(U[:, end-1], Δx, objectives)

    δJ = dot(δU[:, end], Λ[:, end])

    for n in 2:M
        Δt = t[n] - t[n-1]
        Λ_temp .= Ref(zero(eltype(U)))
        OptimalBath.add_objective_source!(Λ_temp, U[:, n-1], Δt, Δx, objectives)
        if n < M
            OptimalBath.DiscreteAdjoints.add_objective_timestep_source!(Λ_temp, U[:, n-1], J_final, Δx, 0.25, objectives)
        end
        δJ += dot(δU[:, n-1], Λ_temp)
    end
    return δJ
end

function general_adjoint_dot_test(N, bathymetry)
    U0 = rand(State{Float64}, N)
    β = zeros(N+1)
    δU0 = rand(State{Float64}, N)
    U, δU, t, problem = tangent_linear_model(N, U0, δU0, bathymetry)

    U = unsafe_to_depth!(U, bathymetry)
    β = zeros(N+1)
    objectives = Objectives(interior_objective=KineticEnergy())
    Λ0 = rand(State{Float64}, N)
    Δx = 1/N
    Λ = solve_adjoint(Λ0, U, objectives, bathymetry, t, Δx, DiscreteAdjointSWE(problem))

    adjoint_dot_product_test = dot(δU[:, 1], Λ[:, 1])

    @test adjoint_dot_product_test ≈ compute_perturbation_with_nonzero_objective(Λ, U, δU, Δx, t, objectives)
end

@testset "Adjoint dot product test" begin
    adjoint_dot_test(8, zeros(9))
    adjoint_dot_test(8, -rand(9))
    general_adjoint_dot_test(8, zeros(9))
    general_adjoint_dot_test(8, -rand(9))
end

function compare_to_ad(N, initial_bathymetry, β)
    U0 = rand(State{Float64}, N)
    U0 = States{Average, Elevation}(U0)

    problem = SinFVMPrimalSWEProblem(N, U0, 0.1, reconstruction=NoReconstruction(), timestepper=ForwardEuler(); initial_bathymetry=initial_bathymetry)
    objectives = Objectives(
        interior_objective=KineticEnergy(),
        terminal_objective=KineticEnergy(),
    )
    forward_ad = ForwardADGradient(β)
    discrete_adjoint = DiscreteAdjointGradient()
    da_objective, da_gradient = compute_objective_and_gradient(β, problem, objectives, discrete_adjoint)
    forward_ad_objective, forward_ad_gradient = compute_objective_and_gradient(β, problem, objectives, forward_ad)

    @test da_objective ≈ forward_ad_objective
    @test forward_ad_gradient ≈ da_gradient
end

@testset "Compare to AD" begin
    N = 10
    compare_to_ad(N, zeros(N+1), zeros(N+1))
    compare_to_ad(N, -rand(N+1), rand(N+1))
end