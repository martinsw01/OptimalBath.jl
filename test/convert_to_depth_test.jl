@testset "Test promotion" begin
    N, M = 3, 4
    U = States{Average, Elevation}(fill(State(1, 2), N, M))
    b = ones(N + 1)

    @assert eltype(U.U) === State{Int64}
    @assert eltype(b) === Float64

    V = to_depth(U, b)
    @test eltype(V.U) === State{Float64}
end

@testset "Test in-place conversion to depth" begin
    w = 2.0
    N, M = 3, 4
    U_inner = fill(State(w, 3), N, M)
    U = States{Average, Elevation}(U_inner)
    b = ones(N + 1)
    V = States{Average, Depth}(similar(U_inner))
    to_depth!(V, U, b)

    U_inner_expected = fill(State(w - 1.0, 3), N, M)

    @test V.U == U_inner_expected
end

@testset "Test not posissible to convert depth to depth" begin
    N, M = 3, 4
    U = States{Average, Depth}(fill(State(1, 2), N, M))
    b = ones(N + 1)

    @test_throws MethodError to_depth(U, b)
    @test_throws MethodError to_depth!(U, U, b)
    @test_throws MethodError unsafe_to_depth!(U, b)
end

@testset "Test left and right conversion" begin
    N, M = 3, 4
    U_left = States{Left, Elevation}(fill(State(3, 2), N, M))
    U_right = States{Right, Elevation}(fill(State(5, 2), N, M))
    b = rand(N + 1)
    U_depth_left = to_depth(U_left, b)
    U_depth_right = to_depth(U_right, b)

    depth_left_expected = repeat(3 .- b[1:end-1], outer=(1, M))
    depth_right_expected = repeat(5 .- b[2:end], outer=(1, M))
    @test height.(U_depth_left.U) == depth_left_expected
    @test height.(U_depth_right.U) == depth_right_expected
end

@testset "Test average conversion" begin
    N, M = 3, 4
    U = States{Average, Elevation}(fill(State(4, 2), N, M))
    b = rand(N + 1)
    U_depth = to_depth(U, b)
    depth_expected = repeat(4 .- (b[1:end-1] .+ b[2:end]) ./ 2, outer=(1, M))
    @test height.(U_depth.U) == depth_expected
end