using StatsBase: sample

function create_random_time_array(T, M)
    t = rand(M-2) * T
    sort!(t)
    return [0.; t; T]
end


@testset "Test constant mass" begin
    N, M = 5, 6
    h, hu, L, T = rand(4)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = States{Average, Depth}(fill(State(h, hu), N, M))
    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Mass())
    objective = OptimalBath.compute_objective(U, t, step(x), β, objectives, ForwardEuler)
    expected_objective = h * L * T
    @test objective ≈ expected_objective atol=1e-6
end

@testset "Test objective_indices" begin

    N, M = 5, 6
    h, hu, L, T = rand(4)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = States{Average, Depth}(fill(State(h, hu), N, M))

    number_of_objective_cells = rand(2:N)
    objective_indices = sample(1:N, number_of_objective_cells; replace=false)

    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Mass(), objective_indices=objective_indices)
    objective = OptimalBath.compute_objective(U, t, step(x), β, objectives, ForwardEuler)
    expected_objective = h * L * T * (number_of_objective_cells / N)
    @test objective ≈ expected_objective atol=1e-6
end


@testset "Test affine in space mass" begin
    N, M = 5, 6
    T = rand()
    L = N - 1
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = States{Average, Depth}([State(xj, rand()) for xj in 0:L, _ in 1:M])
    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Mass())
    objective = OptimalBath.compute_objective(U, t, step(x), β, objectives, ForwardEuler)
    expected_objective = 0.5 * L^2 * T
    @test objective ≈ expected_objective
end

@testset "Test 2D affine in space mass" begin
    Nx, Ny, M = 5, 6, 7
    L, T = rand(2)
    x = range(0, stop=L, length=Nx+1)
    x_centers = x[1:end-1] .+ step(x)/2
    y = range(0, stop=L, length=Ny+1)
    y_centers = y[1:end-1] .+ step(y)/2
    t = create_random_time_array(T, M)
    β = ones(Nx+1, Ny+1)
    U = States{Average, Depth}([State(xj + yk, rand(), rand())
                                for xj in x_centers,
                                    yk in y_centers,
                                    _ in 1:M])
    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Mass(),
                                        terminal_objective=OptimalBath.Mass(),
                                        objective_indices=CartesianIndices((Nx, Ny)),
                                        design_indices=CartesianIndices(β))
    objective = OptimalBath.compute_objective(U, t, (step(x), step(y)), β, objectives, ForwardEuler)
    expected_objective = L^3 * (T + 1)
    @test objective ≈ expected_objective
end



@testset "Test affine in time mass" begin
    N, M = 5, 6
    h0, L, T = rand(3)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = States{Average, Depth}([State(h0 * tn, rand()) for _ in 1:N, tn in t])
    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Mass())
    objective = OptimalBath.compute_objective(U, t, step(x), β, objectives, RK2)
    expected_objective = 0.5 * h0 * L * T^2
    @test objective ≈ expected_objective atol=1e-6
end

@testset "Test 2D affine in time mass" begin
    Nx, Ny, M = 5, 6, 7
    h0, Lx, Ly, T = rand(4)
    x = range(0, stop=Lx, length=Nx+1)
    y = range(0, stop=Ly, length=Ny+1)
    t = create_random_time_array(T, M)
    β = ones(Nx+1, Ny+1)
    U = States{Average, Depth}([State(h0 * tn, rand(), rand())
                                for _ in 1:Nx,
                                    _ in 1:Ny,
                                    tn in t])
    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Mass(),
                                        design_indices=CartesianIndices(β),
                                        objective_indices=CartesianIndices((Nx, Ny)))
    objective = OptimalBath.compute_objective(U, t, (step(x), step(y)), β, objectives, RK2)
    expected_objective = 0.5 * h0 * Lx * Ly * T^2
    @test objective ≈ expected_objective
end


@testset "Test affine in space and time mass" begin
    N, M = 5, 6
    h0, L, T = rand(3)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    Ul = [State(h0 * x[j] * tn, rand()) for j in 1:N, tn in t]
    Ur = [State(h0 * x[j] * tn, rand()) for j in 2:(N+1), tn in t]
    Ul = States{Left, Depth}(Ul)
    Ur = States{Right, Depth}(Ur)
    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Mass())
    objective = OptimalBath.compute_objective(Ul, Ur, t, step(x), β, objectives, RK2)
    expected_objective = 0.25 * h0 * L^2 * T^2
    @test objective ≈ expected_objective atol=1e-6
end


@testset "Test no-reconstruction equals average" begin
    N, M = 5, 6
    L, T = rand(2)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = rand(N+1)
    U = [rand(State{2, Float64}) for _ in 1:N, _ in 1:M]
    U = States{Average, Depth}(U)
    Ul = States{Left, Depth}(U.U)
    Ur = States{Right, Depth}(U.U)
    objectives = OptimalBath.Objectives(interior_objective=OptimalBath.Energy(),
                                        terminal_objective=OptimalBath.Mass(),
                                        objective_indices=3:N,
                                        design_indices=1:N-1,
                                        regularization=β -> sum(β.^2))
    objective_average = OptimalBath.compute_objective(U, t, step(x), β, objectives, ForwardEuler)
    objective_reconstructed = OptimalBath.compute_objective(Ul, Ur, t, step(x), β, objectives, ForwardEuler)
    @test objective_average ≈ objective_reconstructed
end


@testset "Test affine terminal objective" begin
    N, M = 5, 6
    h, L, T = rand(3)
    x = range(0, stop=L, length=N+1)
    t = create_random_time_array(T, M)
    β = ones(N+1)
    U = States{Average, Depth}(fill(State(rand(), rand()), N, M))
    U.U[:, end] .= (State(h * (j/(N-1)), rand()) for j in 0:(N-1))


    objectives = OptimalBath.Objectives(terminal_objective=OptimalBath.Mass())
    objective = OptimalBath.compute_objective(U, t, step(x), β, objectives, ForwardEuler)
    expected_objective = 0.5 * h * L
    @test objective ≈ expected_objective atol=1e-6
end