export WellBalancedNoReconstruction

struct WellBalancedNoReconstruction <: WellBalancedReconstruction end

function reconstruction_buffers(U, ::Reconstruction)
    U_left = similar(U)
    U_right = similar(U)
    return U_left, U_right
end

function set_depth(W::State, depth)
    return setindex(W, depth, 1)
end

function compute_depths(W::State, b_left, b_right)
    h_left = height(W) - b_left
    h_right = height(W) - b_right

    if h_left < 0
        h_right += h_left
        h_left = zero(h_left)
    elseif h_right < 0
        h_left += h_right
        h_right = zero(h_right)
    end

    return h_left, h_right
end

function reconstruct!(out_left, out_right, W, b, dir, grid::Grid, ::WellBalancedNoReconstruction)
    for_each_cell(grid) do j
        b_left = b_at(Left, b, j, dir)
        b_right = b_at(Right, b, j, dir)

        h_left, h_right = compute_depths(W[j], b_left, b_right)

        out_left[j] = set_depth(W[j], h_left)
        out_right[j] = set_depth(W[j], h_right)
    end
end