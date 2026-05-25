function reconstruction_buffers(U, ::NoReconstruction)
    U_left = similar(U)
    U_right = U_left
    return U_left, U_right
end

function subtract_bathymetry(W::State{2}, b)
    return State(height(W) - b, momentum(W, XDIR))
end

function subtract_bathymetry(W::State{3}, b)
    return State(height(W) - b, momentum(W, XDIR), momentum(W, YDIR))
end

function reconstruct!(U_left, U_right, W, b, dir, grid, ::NoReconstruction)
    @assert U_left === U_right "NoReconstruction assumes that U_left and U_right are the same array"
    for_each_cell(grid) do j
        b_j = b_at(Average, b, j)
        U = subtract_bathymetry(W[j], b_j)
        U_left[j] = U
    end
end