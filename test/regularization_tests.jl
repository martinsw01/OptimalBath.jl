@testset "Test vector space" begin
    r1 = SoftL1(20.)
    r2 = L2()
    a, b = rand(2)

    β = rand(10)
    
    @test r1(β) + r2(β) ≈ (r1 + r2)(β)
    @test a * r1(β) ≈ (a * r1)(β)
    @test a * r1(β) + b * r2(β) ≈ (a * r1 + b * r2)(β)
    @test r1(β) / a ≈ (r1 / a)(β)
    @test r1(β) - r2(β) ≈ (r1 - r2)(β)

    @test "$r1 + $r2" == "SoftL1(20.0) + L2()"
    @test "$a * $r1 + $b * $r2" == "$a * SoftL1(20.0) + $b * L2()"
end

function compare_to_AD(regularizations, β)
    for regularization in regularizations
        @testset "Test $regularization gradient" begin
            g = similar(β)
            g_expected = zero(β)

            gradient!(g, β, regularization)
            add_gradient!(g_expected, β, 1, x -> regularization(x))

            @test g ≈ g_expected
        end
    end
end

@testset "Test gradients" begin
    reg_atoms = [NoRegularization(), L2(), SoftL1(rand()), SoftTV(rand())]
    reg_linear_combinations = [rand() * r1 + rand() * r2 for r1 in reg_atoms for r2 in reg_atoms]
    compare_to_AD(reg_atoms, rand(10))
    compare_to_AD(reg_linear_combinations, rand(10))
end