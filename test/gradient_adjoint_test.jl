using Random

@testset "Test constant primary and adjoint" begin
    h, hu, λ1, λ2 = rand(4)
    g = 9.81
    N = 10
    M = 6

    U = fill(State(h, hu), (N, M))
    U = States{Average, Depth}(U)
    Λ = fill(State(λ1, λ2), (N, M))

    t = [0; cumsum(rand(M-1))]
    expected_gradient = zeros(N + 1)

    expected_gradient[[1, end]] .= h*λ2 * g .* [-1, 1] * t[end]

    gradient = similar(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, :)

    @test gradient ≈ expected_gradient atol=1e-14

    OptimalBath.compute_gradient!(gradient, Λ, U, t, eachindex(gradient))

    @test gradient ≈ expected_gradient atol=1e-14
end

# @testset "Test affine in time primary and constant adjoint (fails)" begin
#     h, hu, λ1, λ2 = rand(4)

#     g = 9.81
#     N = 10
#     M = 6

#     t = [0; cumsum(rand(M-1))]
#     U = [[h * t[n], hu] for _ in 1:N, n in 1:M]
#     Λ = fill([λ1, λ2], (N, M))

#     expected_gradient = zeros(N + 1)
#     expected_gradient[[1, end]] .= 0.5 * h*λ2 * g .* [-1, 1] * t[end]^2

#     gradient = similar(expected_gradient)
#     OptimalBath.compute_gradient!(gradient, Λ, U, t, :)

#     # Second order accuracy in time not implemented, so this will fail
#     @test gradient ≈ expected_gradient atol=1e-6
# end

@testset "Test piecewise primary and constant adjoint" begin
    h, hu, λ1, λ2 = rand(4)
    g = 9.81
    N = 6
    M = 4

    t = [0; cumsum(rand(M-1))]
    U = [State(h * mod(j, 2), hu) for j in 1:N, _ in 1:M]
    U = States{Average, Depth}(U)
    Λ = fill(State(λ1, λ2), (N, M))

    expected_gradient = h * λ2 .* [-1, 1, -1, 1, -1, 1, 0] * g * t[end]

    gradient = similar(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, :)

    @test gradient ≈ expected_gradient atol=1e-6

    OptimalBath.compute_gradient!(gradient, Λ, U, t, eachindex(gradient))

    @test gradient ≈ expected_gradient atol=1e-6
end

@testset "Test p.w. affine in space primary and constant adjoint" begin
    hl, hr, hu, λ1, λ2 = rand(5)
    g = 9.81
    N = 10
    M = 6
    t = [0; cumsum(rand(M-1))]
    x = range(0, stop=1, length=N+1)
    Ul = [State(hl, hu) for xj in x[1:N], _ in 1:M]
    Ur = [State(hr, hu) for xj in x[1:N], _ in 1:M]
    Ul = States{Left, Depth}(Ul)
    Ur = States{Right, Depth}(Ur)
    Λ = fill(State(λ1, λ2), (N, M))

    expected_gradient = 0.5 * g * λ2 * (hr + hl) * t[end] .* [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    gradient = similar(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, Ul, Ur, t, :)

    @test gradient ≈ expected_gradient atol=1e-6

    OptimalBath.compute_gradient!(gradient, Λ, Ul, Ur, t, eachindex(gradient))
    @test gradient ≈ expected_gradient atol=1e-6
end



@testset "Test selected design parameters" begin
    h, hu, λ1, λ2 = rand(4)

    g = 9.81
    N = 10
    M = 6

    U = fill(State(h, hu), (N, M))
    U = States{Average, Depth}(U)
    Λ = fill(State(λ1, λ2), (N, M))

    t = [0; cumsum(rand(M-1))]
    expected_gradient = zeros(2)

    expected_gradient[1] = -h*λ2 * g * t[end]

    gradient = zero(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, [1 2])

    @test gradient ≈ expected_gradient atol=1e-14


    expected_gradient = zeros(4)
    expected_gradient[end] = h*λ2 * g * t[end]
    gradient = zero(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, [2 5 6 N + 1])

    @test gradient ≈ expected_gradient atol=1e-14
end
