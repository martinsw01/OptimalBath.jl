export WellBalancedNoReconstruction

struct WellBalancedNoReconstruction <: WellBalancedReconstruction end

function reconstruction_buffers(U, ::Reconstruction)
    U_left = similar(U)
    U_right = similar(U)
    return U_left, U_right
end

function reconstruct_side(W::State{2}, b_side)
    h_side = height(W) - b_side
    if h_side < 0
        return zero(W)
    else
        return State(h_side, momentum(W, XDIR))
    end
end

function reconstruct_side(W::State{3}, b_side)
    h_side = height(W) - b_side
    if h_side < 0
        return zero(W)
    else
        return State(h_side, momentum(W, XDIR), momentum(W, YDIR))
    end
end

# TODO: this works poorly in dry states, as it creates water on one of the sides. Replace for example with a slop-adjustment.
function reconstruct!(out_left, out_right, W, b, dir, grid::Grid, ::WellBalancedNoReconstruction)
    for_each_cell(grid) do j
        b_left = b_at(Left, b, j, dir)
        b_right = b_at(Right, b, j, dir)

        out_left[j] = reconstruct_side(W[j], b_left)
        out_right[j] = reconstruct_side(W[j], b_right)
    end
end