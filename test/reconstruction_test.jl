function forward_slope(Ul, Uc, Ur)
    return Ur .- Uc
end

# function convert_to_depth(U, b)
#     h, p = U
#     return State(h - b, p)
# end

# function is_dry(U, b, ε)
#     return abs(height(U) - b) < ε
# end

# function is_interface_dry(U, b, ε)
#     return height(U) < b + ε
# end

# function reconstruct_cell(Ul, Uc, Ur, B_left, B_right, ε, compute_slope)
#     if is_dry(Uc, 0.5 * (B_left + B_right), ε)
#         return zero(Ul), zero(Ur)
#     end
#     if is_interface_dry(Uc, B_left, ε)
#         return zero(Ul), convert_to_depth(Uc, B_right)
#     end
#     if is_interface_dry(Uc, B_right, ε)
#         return convert_to_depth(Uc, B_left), zero(Ur)
#     end

#     slope = compute_slope(Ul, Uc, Ur)

#     U_left = Uc .- 0.5 .* slope
#     U_right = Uc .+ 0.5 .* slope

#     return convert_to_depth(U_left, B_left), convert_to_depth(U_right, B_right)
# end

depth_cutoff = 1e-6


@testset "Test all wet lake-at-rest, constant bathymetry" begin
    h, p = rand(2)
    bl = br = -rand()
    Ul = Uc = Ur = State(h, p)

    Uc_left, Uc_right = OptimalBath.reconstruct_cell(Ul, Uc, Ur, bl, br, depth_cutoff, forward_slope)
    @test Uc_left == State(h - bl, p)
    @test Uc_right == State(h - br, p)
end

@testset "Test all wet lake-at-rest, varying bathymetry" begin
    h, p = rand(2)
    bl = -rand()
    br = -rand()
    Ul = Uc = Ur = State(h, p)

    Uc_left, Uc_right = OptimalBath.reconstruct_cell(Ul, Uc, Ur, bl, br, depth_cutoff, forward_slope)
    @test Uc_left == State(h - bl, p)
    @test Uc_right == State(h - br, p)
end

@testset "Test constant bathymetry, varying water heights" begin
    p = rand()
    bl = br = -rand()
    hc = 1 + rand()
    hl = hc - 1
    hr = hc + 1
    Ul = State(hl, p)
    Uc = State(hc, p)
    Ur = State(hr, p)

    Uc_left, Uc_right = OptimalBath.reconstruct_cell(Ul, Uc, Ur, bl, br, depth_cutoff, forward_slope)
    @test Uc_left == State(hc - 0.5 - bl, p)
    @test Uc_right == State(hc + 0.5 - br, p)
end

@testset "Test partly wet lake-at-rest, below average" begin
    bl = 1
    br = 2.5
    Ul = State(1.5, 1)
    Uc = State(1.5, 1)
    Ur = State(2.5, 1)

    Uc_left, Uc_right = OptimalBath.reconstruct_cell(Ul, Uc, Ur, bl, br, depth_cutoff, forward_slope)
    @test Uc_left == State(0.5, 1)
    @test Uc_right == State(0, 0)
end

@testset "Test partly wet lake-at-rest, above average" begin
    bl = 1
    br = 2.5
    Ul = State(2., 1)
    Uc = State(2., 1)
    Ur = State(2.5, 1)

    Uc_left, Uc_right = OptimalBath.reconstruct_cell(Ul, Uc, Ur, bl, br, depth_cutoff, forward_slope)
    @test Uc_left == State(1., 1)
    @test Uc_right == State(0., 0.)
end

@testset "Test potentially negative reconstruction" begin
    # Create a scenario where a naive reconstruction would yield negative water heights, but the bathymetry handling should cap it to zero.
    bl = 1
    br = 0
    Ul = State(2., 1)
    Uc = State(1.1, 1)
    Ur = State(1.6, 1)

    Uc_left, Uc_right = OptimalBath.reconstruct_cell(Ul, Uc, Ur, bl, br, depth_cutoff, forward_slope)
    @test Uc_left == State(0., 0.)
    @test Uc_right == State(1.35, 1.)
end