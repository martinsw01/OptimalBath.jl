@testset "Test 1D selection" begin
    b = rand(4)
    @test b_at(Left, b, 2, XDIR) ≈ b[2]
    @test b_at(Right, b, 2, XDIR) ≈ b[3]
    @test b_at(Average, b, 2, XDIR) ≈ 0.5*(b[2] + b[3])

    @test b_at(Left, b, CartesianIndex(2), XDIR) ≈ b[2]
    @test b_at(Right, b, CartesianIndex(2), XDIR) ≈ b[3]
    @test b_at(Average, b, CartesianIndex(2), XDIR) ≈ 0.5*(b[2] + b[3])
end

@testset "Test 2D selection" begin
    b = rand(4, 5)
    @test b_at(Left, b, 2, 3, XDIR) ≈ 0.5*(b[2, 3] + b[2, 4])
    @test b_at(Right, b, 2, 3, XDIR) ≈ 0.5*(b[3, 3] + b[3, 4])
    @test b_at(Left, b, 2, 3, YDIR) ≈ 0.5*(b[2, 3] + b[3, 3])
    @test b_at(Right, b, 2, 3, YDIR) ≈ 0.5*(b[2, 4] + b[3, 4])
    @test b_at(Average, b, 2, 3) ≈ 0.25*(b[2, 3] + b[2, 4] + b[3, 3] + b[3, 4])

    @test b_at(Left, b, CartesianIndex(2, 3), XDIR) ≈ 0.5*(b[2, 3] + b[2, 4])
    @test b_at(Right, b, CartesianIndex(2, 3), XDIR) ≈ 0.5*(b[3, 3] + b[3, 4])
    @test b_at(Left, b, CartesianIndex(2, 3), YDIR) ≈ 0.5*(b[2, 3] + b[3, 3])
    @test b_at(Right, b, CartesianIndex(2, 3), YDIR) ≈ 0.5*(b[2, 4] + b[3, 4])
    @test b_at(Average, b, CartesianIndex(2, 3)) ≈ 0.25*(b[2, 3] + b[2, 4] + b[3, 3] + b[3, 4])
end