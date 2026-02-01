using Random

@testset "Test constant primary and adjoint" begin
    h, hu, λ1, λ2 = rand(4)
    g = 9.81
    N = 10
    M = 6

    U = fill(State(h, hu), (N, M))
    U = States{Average, Depth}(U)
    Λ = fill([λ1, λ2], (N, M))

    t = [0; cumsum(rand(M-1))]
    expected_gradient = zeros(N + 1)

    expected_gradient[[1, end]] .= h*λ2 * g .* [-1, 1] * t[end]

    gradient = similar(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, nothing, :)

    @test gradient ≈ expected_gradient atol=1e-14

    OptimalBath.compute_gradient!(gradient, Λ, U, t, nothing, eachindex(gradient))

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
#     OptimalBath.compute_gradient!(gradient, Λ, U, t, nothing, :)

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
    Λ = fill([λ1, λ2], (N, M))

    expected_gradient = h * λ2 .* [-1, 1, -1, 1, -1, 1, 0] * g * t[end]

    gradient = similar(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, nothing, :)

    @test gradient ≈ expected_gradient atol=1e-6

    OptimalBath.compute_gradient!(gradient, Λ, U, t, nothing, eachindex(gradient))

    @test gradient ≈ expected_gradient atol=1e-6
end

@testset "Test selected design parameters" begin
    h, hu, λ1, λ2 = rand(4)

    g = 9.81
    N = 10
    M = 6

    U = fill(State(h, hu), (N, M))
    U = States{Average, Depth}(U)
    Λ = fill([λ1, λ2], (N, M))

    t = [0; cumsum(rand(M-1))]
    expected_gradient = zeros(2)

    expected_gradient[1] = -h*λ2 * g * t[end]

    gradient = zero(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, nothing, [1 2])

    @test gradient ≈ expected_gradient atol=1e-14


    expected_gradient = zeros(4)
    expected_gradient[end] = h*λ2 * g * t[end]
    gradient = zero(expected_gradient)
    OptimalBath.compute_gradient!(gradient, Λ, U, t, nothing, [2 5 6 N + 1])

    @test gradient ≈ expected_gradient atol=1e-14
end
