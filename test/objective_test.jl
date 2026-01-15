function create_random_time_array(T, M)
    t = rand(M-2) * T
    sort!(t)
    return [0.; t; T]
end

no_parameter_objective(β) = 0.0

@testset "Test constant mass" begin
    N, M = 5, 6
    h, hu, L, T = rand(4)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = fill([h, hu], (N, M))
    interior_objective_density = OptimalBath.Mass()
    terminal_objective_density = OptimalBath.NoObjective()
    objective = OptimalBath.compute_objective(U, t, x, β, interior_objective_density, terminal_objective_density, no_parameter_objective)
    expected_objective = h * L * T
    @test objective ≈ expected_objective atol=1e-6
end


@testset "Test affine in space mass" begin
    N, M = 5, 6
    T = rand()
    L = N - 1
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = [[xj, rand()] for xj in 0:L, _ in 1:M]
    interior_objective_density = OptimalBath.Mass()
    terminal_objective_density = OptimalBath.NoObjective()
    objective = OptimalBath.compute_objective(U, t, x, β, interior_objective_density, terminal_objective_density, no_parameter_objective)
    expected_objective = 0.5 * L^2 * T
    @test objective ≈ expected_objective atol=1e-6
end


@testset "Test affine in time mass" begin
    N, M = 5, 6
    h0, L, T = rand(3)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = [[h0 * tn, rand()] for _ in 1:N, tn in t]
    interior_objective_density = OptimalBath.Mass()
    terminal_objective_density = OptimalBath.NoObjective()
    objective = OptimalBath.compute_objective(U, t, x, β, interior_objective_density, terminal_objective_density, no_parameter_objective)
    expected_objective = 0.5 * h0 * L * T^2
    @test objective ≈ expected_objective atol=1e-6
end


@testset "Test affine terminal objective" begin
    N, M = 5, 6
    h, L, T = rand(3)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = fill([rand(), rand()], (N, M))
    U[:, end] .= ([h * (j/(N-1)), rand()] for j in 0:(N-1))

    interior_objective_density = OptimalBath.NoObjective()
    terminal_objective_density = OptimalBath.Mass()
    objective = OptimalBath.compute_objective(U, t, x, β, interior_objective_density, terminal_objective_density, no_parameter_objective)
    expected_objective = 0.5 * h * L
    @test objective ≈ expected_objective atol=1e-6
end