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

function tangent_linear_model(N, U0, δU0)
    function f(U)
        β = zeros(eltype(eltype(U)), N+1)
        U = States{Average, Elevation}(unflatten(U))
        problem = SinFVMPrimalSWEProblem(N, U, 0.1, reconstruction=NoReconstruction(), timestepper=ForwardEuler())
        U, t, x = solve_primal(problem, β)
        return U, t, x, problem
    end
    function g(U)
        return flatten(f(U)[1].U)
    end

    J_flat = ForwardDiff.jacobian(g, flatten(U0))

    U, t, x, problem = f(flatten(U0))

    δU = unflatten(J_flat * flatten(δU0), N)
    return U.U, δU, t, problem
end

using LinearAlgebra: dot
function adjoint_dot_test(adjoint_type)
    N = 5

    U0 = rand(State{Float64}, N)
    δU0 = rand(State{Float64}, N)
    U, δU, t, problem = tangent_linear_model(N, U0, δU0)

    U = States{Average, Depth}(U)
    β = zeros(N+1)

    dJdU = zero(U.U)
    Λ0 = rand(State{Float64}, N)
    Δx = 1/N

    Λ = solve_adjoint(Λ0, U, dJdU, β, t, Δx, adjoint_type(problem))

    adjoint_dot_product_test = dot(δU[:, end], Λ0)
    @test adjoint_dot_product_test ≈ dot(δU0, Λ[:, 1])

    # dot_products = [dot(δU[:, n], Λ[:,n]) for n in 1:size(U.U, 2)]
    # all_equal = reduce(dot_products, init=true) do acc, dp
    #     return acc && dp ≈ adjoint_dot_product_test
    # end
    # @test all_equal
    # reduce(hcat, Λ[:, 1])
end

@testset "Compare with AD" begin
    adjoint_dot_test(DiscreteAdjoint)
end