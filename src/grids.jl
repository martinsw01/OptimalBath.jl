export Grid, Grid1D, Grid2D
export get_Δx
export for_each_cell, for_each_interior_directional_stencil, for_each_left_boundary_directional_stencil, for_each_right_boundary_directional_stencil

struct Grid{Dims, FloatType, Domain}
    Δx::NTuple{Dims, FloatType}
    N::NTuple{Dims, Int64}
    domain::Domain
end

function get_Δx(grid::Grid{1}, dir::Val{1})
    return grid.Δx[1]
end

function get_Δx(grid::Grid{2}, ::Val{dir}) where dir
    return grid.Δx[dir]
end

function Grid1D(Nx; domain=[0.0 1.0])
    Δx = (domain[2] - domain[1]) / Nx
    Δ = (Δx,)
    N = (Nx,)
    return Grid{1, eltype(Δ), typeof(domain)}(Δ, N, domain)
end

function Grid2D(Nx, Ny; domain=[0.0 1.0; 0.0 1.0])
    Δx = (domain[1, 2] - domain[1, 1]) / Nx
    Δy = (domain[2, 2] - domain[2, 1]) / Ny
    Δ = (Δx, Δy)
    N = (Nx, Ny)
    return Grid{2, eltype(Δ), typeof(domain)}(Δ, N, domain)
end

function interior_cell_indices(grid::Grid{1})
    return 2:grid.N[1]-1
end

function for_each_interior_directional_stencil(f, dir, grid::Grid{1})
    for i in interior_cell_indices(grid)
        f(i-1, i, i+1)
    end
end

function for_each_cell(f, grid::Grid{1})
    indices = 1:grid.N[1]
    for i in indices
        f(i)
    end
end

function for_each_left_boundary_directional_stencil(f, dir, ::Grid{1})
    f(1, 2)
end

function for_each_right_boundary_directional_stencil(f, dir, grid::Grid{1})
    N = grid.N[1]
    f(N-1, N)
end
