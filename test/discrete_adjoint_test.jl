function flatten(U::AbstractMatrix)
    return reduce(vcat, vec(U))
end

function flatten(U::AbstractVector)
    return reduce(vcat, U)
end

function unflatten(u_flat)
    N = length(u_flat) ÷ 2
    [State{2}(u_flat[2i-1:2i]) for i in 1:N]
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
        problem = PrimalSWEProblem(N, U, 0.1, initial_bathymetry=bathymetry)

        solver = VolumeFluxesSolver(problem, SolverOptions(), eltype(eltype(U.U)))
        U, t, x = solve_primal(solver, β)
        return U, t, x, solver
    end
    function g(U)
        return flatten(f(U)[1].U)
    end

    J_flat = ForwardDiff.jacobian(g, flatten(U0))

    U, t, x, solver = f(flatten(U0))

    δU = unflatten(J_flat * flatten(δU0), N)
    return U, δU, t, solver
end

function display_jacobians(J, N)
    for i in 1:2N:size(J, 1)-2N
        J_prev_inv = inv(J[i:i+2N-1, :])
        J_next = J[i+2N:i+4N-1, :]
        display((J_prev_inv * J_next)')
    end
end

function dot(a, b)
    return a' * b
end

function adjoint_dot_test(N, compute_U0, bathymetry)
    U0 = compute_U0(N)
    δU0 = rand(State{2, Float64}, N)
    U, δU, t, solver = tangent_linear_model(N, U0.U, δU0, bathymetry)

    U = unsafe_to_depth!(U, bathymetry)
    β = zeros(N+1)

    objectives = Objectives()
    Λ0 = rand(State{2, Float64}, N)
    Δx = 1/N

    Λ = solve_adjoint(Λ0, U, objectives, bathymetry, t, Δx, DiscreteAdjointSWE(solver))

    adjoint_dot_product_test = dot(δU[:, end], Λ0)
    @test adjoint_dot_product_test ≈ dot(δU0, Λ[:, 1])
end

function compute_perturbation_with_nonzero_objective(Λ, U, δU, Δx, t, objectives, da)
    U = U.U
    N, M = size(U)
    Λ_temp = similar(U, N)

    J_final = OptimalBath.DiscreteAdjoints.compute_objective_step(U[:, end-1], Δx, objectives)

    δJ = dot(δU[:, end], Λ[:, end])

    for n in 2:M
        Δt = t[n] - t[n-1]
        fill!(Λ_temp, zero(eltype(U)))
        OptimalBath.add_objective_source!(Λ_temp, U[:, n-1], Δt, Δx, objectives)
        if n < M
            OptimalBath.DiscreteAdjoints.add_objective_timestep_source!(Λ_temp, U[:, n-1], J_final, 0.25, objectives, da)
        end
        δJ += dot(δU[:, n-1], Λ_temp)
    end
    return δJ
end

function general_adjoint_dot_test(N, compute_U0, bathymetry)
    U0 = compute_U0(N)
    β = zeros(N+1)
    δU0 = rand(State{2, Float64}, N)
    U, δU, t, solver = tangent_linear_model(N, U0.U, δU0, bathymetry)

    U = unsafe_to_depth!(U, bathymetry)
    β = zeros(N+1)
    objectives = Objectives(interior_objective=KineticEnergy())
    Λ0 = rand(State{2, Float64}, N)
    Δx = 1/N
    da = DiscreteAdjointSWE(solver)
    Λ = solve_adjoint(Λ0, U, objectives, bathymetry, t, Δx, da)

    adjoint_dot_product_test = dot(δU[:, 1], Λ[:, 1])

    @test adjoint_dot_product_test ≈ compute_perturbation_with_nonzero_objective(Λ, U, δU, Δx, t, objectives, da)
end


function compare_to_ad(N, compute_U0, initial_bathymetry, β)
    U0 = compute_U0(N)

    problem = PrimalSWEProblem(N, U0, 0.1, initial_bathymetry=initial_bathymetry)
    spec = SolverSpec(problem, VolumeFluxesBackend())
    objectives = Objectives(
        interior_objective=KineticEnergy(),
        terminal_objective=KineticEnergy(),
    )
    forward_ad = ForwardADGradient(β)
    discrete_adjoint = DiscreteAdjointGradient()
    da_objective, da_gradient = compute_objective_and_gradient(β, spec, objectives, discrete_adjoint)
    forward_ad_objective, forward_ad_gradient = compute_objective_and_gradient(β, spec, objectives, forward_ad)
    @test da_objective ≈ forward_ad_objective
    @test forward_ad_gradient ≈ da_gradient
end

function compare_to_2D_ad(N, compute_U0, initial_bathymetry, β)
    U0 = compute_U0(N)
    grid = Grid2D(N)
    problem = PrimalSWEProblem(U0, 0.1, grid, initial_bathymetry)
    spec = SolverSpec(problem, VolumeFluxesBackend())
    objectives = Objectives(
        interior_objective=KineticEnergy(),
        terminal_objective=KineticEnergy(),
        objective_indices=CartesianIndices(N),
        design_indices=CartesianIndices(N .+ 1),
    )
    forward_ad = ForwardADGradient(β)
    discrete_adjoint = DiscreteAdjointGradient()
    da_objective, da_gradient = compute_objective_and_gradient(β, spec, objectives, discrete_adjoint)
    forward_ad_objective, forward_ad_gradient = compute_objective_and_gradient(β, spec, objectives, forward_ad)
    @test da_objective ≈ forward_ad_objective
    @test forward_ad_gradient ≈ da_gradient
end

function random_U0(N)
    dims = length(N)
    U0 = rand(State{dims+1, Float64}, N)
    return States{Average, Elevation}(U0)
end

function post_proc_affected_U0(N)
    U0 = fill(State(0.2, 1.), N)
    U0[1] = State(5e-3, 1.)
    return States{Average, Elevation}(U0)
end

@testset "Adjoint dot product test" begin
    @testset "Zero bathymetry" begin
        adjoint_dot_test(8, random_U0, zeros(9))
    end
    @testset "Random bathymetry" begin
        adjoint_dot_test(8, random_U0, -rand(9))
    end
    @testset "With objective, zero bathymetry" begin
        general_adjoint_dot_test(8, random_U0, zeros(9))
    end
    @testset "With objective, random bathymetry" begin
        general_adjoint_dot_test(8, random_U0, -rand(9))
    end
end

@testset "Compare to AD" begin
    N = 10
    @testset "Zero bathymetry" begin
        compare_to_ad(N, random_U0, zeros(N+1), zeros(N+1))
    end
    @testset "Random bathymetry" begin
        compare_to_ad(N, random_U0, -rand(N+1), rand(N+1))
    end
    @testset "Two dimensions" begin
        N = (5, 7)
        compare_to_2D_ad(N, random_U0, -rand(N .+ 1...), -rand(N .+ 1...))
    end
end

@testset "Test post-processing step" begin
    N = 3
    @testset "Adjoint consistency" begin
        adjoint_dot_test(N, post_proc_affected_U0, zeros(N+1))
    end
    @testset "Compare to AD" begin
        compare_to_ad(N, post_proc_affected_U0, zeros(N+1), zeros(N+1))
        compare_to_ad(N, post_proc_affected_U0, range(0, 0.01, N+1), zeros(N+1))
    end
end

function essentially_1D_problem(problem_1D::PrimalSWEProblem, β_1D, Ny)
    N = (problem_1D.grid.N[1], Ny)
    grid_2D = Grid2D(N; domain=[0. 1.; 0. Ny])
    U0_2D = [State(h, hu, 0.) for (h, hu) in problem_1D.U0.U, _ in 1:Ny]
    U0_2D = States{Average, Elevation}(U0_2D)
    initial_bathymetry_2D = repeat(problem_1D.initial_bathymetry, 1, Ny+1)
    problem_2D = PrimalSWEProblem(U0_2D, problem_1D.T, grid_2D, initial_bathymetry_2D)

    β_2D = repeat(β_1D, 1, Ny+1)
    return problem_2D, β_2D
end

function expected_essentially_1D_gradient(gradient_1D, Ny)
    gradient_2D = repeat(gradient_1D, 1, Ny+1)
    gradient_2D[:,1] .*= 0.5
    gradient_2D[:,end] .*= 0.5
    return gradient_2D
end