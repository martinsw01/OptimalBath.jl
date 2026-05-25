export AverageBottomSource

function add_bottom_source!(Λ, n, Δt, b, dir, grid, ::SimpleBottomSource)
    for_each_cell(grid) do j
        Δb = b_at(Right, b, j, dir) - b_at(Left, b, j, dir)
        Λ[j, n] += compute_bottom_source_term(Λ[j, n+1], Δb, Δt, dir)
    end
end


"""
    AverageBottomSource <: AdjointBottomSource

A bottom source balanced with `OuterCentralDifferenceFlux` when using `NoReconstruction`. Uses the central difference of the average
bottom topography in the left and right cells.`
"""
struct AverageBottomSource <: AdjointBottomSource end

function add_bottom_source!(Λ, n, Δt, b, dir, grid, ::AverageBottomSource)
    for_each_left_boundary_directional_stencil(dir, grid) do center, right
        Δb = compute_Δb(b, center, right)
        Λ[center, n] += compute_bottom_source_term(Λ[center, n+1], Δb, Δt, dir)
    end
    for_each_interior_directional_stencil(dir, grid) do left, center, right
        Δb = compute_Δb(b, left, right)
        Λ[center, n] += compute_bottom_source_term(Λ[center, n+1], Δb, Δt, dir)
    end
    for_each_right_boundary_directional_stencil(dir, grid) do left, center
        Δb = compute_Δb(b, left, center)
        Λ[center, n] += compute_bottom_source_term(Λ[center, n+1], Δb, Δt, dir)
    end
end


function compute_Δb(b, left, right)
    return 0.5 * (b_at(Average, b, right) - b_at(Average, b, left))
end

function compute_bottom_source_term(Λ, Δb, Δt, dir)
    S12 = -9.81 * Δb * Δt
    return setindex(zero(Λ), S12 * momentum(Λ, dir), 1)
end
