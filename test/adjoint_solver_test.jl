using StaticArrays

@testset "Test trivial adjoint" begin
    N = 10
    M = 20

    t = [0; cumsum(rand(M-1))]
    Δx = 1.0
    b = rand(N+1)

    U = rand(SVector{2, Float64}, N, M)
    Λ0 = fill(SVector{2}(0., 0.), N)
    dJdU = zero.(U)

    Λ_expected = fill(SVector{2}(0., 0.), size(U)...)
    Λ = OptimalBath.solve_adjoint(Λ0, U, dJdU, b, t, Δx)

    @test Λ == Λ_expected
end


# @testset "Test adjoint with non-trivial terminal loss" begin
#     h = 1.0
#     hu = 0.0
#     N = 70
#     M = 30

#     Δx = rand()
#     Δt = 0.5 * Δx / sqrt(9.81 * h)
#     t = (0:M) .* Δt

#     b = zeros(N+1)
#     Λ0 = fill(SVector{2}(1.0, 1.0), N)
#     U = [SVector{2}(h, hu) for _ in 1:N, _ in 1:M]
#     dJdU = zero.(U)

#     Λ = OptimalBath.solve_adjoint(Λ0, U, dJdU, b, t, Δx)
#     λ2 = reshape([Λ[i][2] for i in eachindex(Λ)], size(Λ))
#     λ1 = reshape([Λ[i][1] for i in eachindex(Λ)], size(Λ))
#     Λ1 = [sum(col)*Δx for col in eachrow(λ1)]


#     # @show size(Λ)
#     # @show size(λ2)
#     # @show maximum(λ2)
#     # @show minimum(λ2)
#     # @show Λ1
#     # @show sum(λ2[:,end])*Δx

#     @test false
    
# end