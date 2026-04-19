export Grid, Grid1D, Grid2D
export XDIRT, YDIRT, XDIR, YDIR
export get_Δx
export for_each_cell, for_each_interior_directional_stencil, for_each_left_boundary_directional_stencil, for_each_right_boundary_directional_stencil
export for_each_interior_interface, for_each_left_boundary_interface, for_each_right_boundary_interface
export directions

const XDIRT = Val{1}
const YDIRT = Val{2}
const XDIR = Val(1)
const YDIR = Val(2)

include("bathymetries.jl")

struct Grid{Dims, FloatType, Domain}
    Δx::NTuple{Dims, FloatType}
    N::NTuple{Dims, Int64}
    domain::Domain
end

function get_Δx(grid::Grid{1}, dir::XDIRT)
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

function directions(::Grid{D}) where D
    return ntuple(Val, D)
end

Grid2D(N::NTuple{2, Integer}; domain=[0.0 1.0; 0.0 1.0]) = Grid2D(N..., domain=domain)

left_index(i::CartesianIndex, ::Val{dir}) where dir = setindex(i, i[dir] - 1, dir)
right_index(i::CartesianIndex, ::Val{dir}) where dir = setindex(i, i[dir] + 1, dir)


function directional_interior_indices(grid::Grid{D}, ::Val{dir}) where {D, dir}
    ranges = ntuple(Val(D)) do d
        if d == dir
            return 2:grid.N[d]-1
        else
            return 1:grid.N[d]
        end
    end
    return CartesianIndices(ranges)
end

function directional_left_boundary_indices(grid::Grid{D}, ::Val{dir}) where {D, dir}
    ranges = ntuple(Val(D)) do d
        if d == dir
            return 1:1
        else
            return 1:grid.N[d]
        end
    end
    return CartesianIndices(ranges)
end

function directional_right_boundary_indices(grid::Grid{D}, ::Val{dir}) where {D, dir}
    ranges = ntuple(Val(D)) do d
        if d == dir
            return grid.N[dir]:grid.N[dir]
        else
            return 1:grid.N[d]
        end
    end
    return CartesianIndices(ranges)
end

function for_each_cell(f, grid::Grid)
    for i in CartesianIndices(grid.N)
        f(i)
    end
end

function for_each_interior_directional_stencil(f, dir::Val, grid::Grid)
    indices = directional_interior_indices(grid, dir)
    for i in indices
        i_left = left_index(i, dir)
        i_right = right_index(i, dir)
        f(i_left, i, i_right)
    end
end

function for_each_left_boundary_directional_stencil(f, dir::Val, grid::Grid)
    indices = directional_left_boundary_indices(grid, dir)
    for i in indices
        i_right = right_index(i, dir)
        f(i, i_right)
    end
end

function for_each_right_boundary_directional_stencil(f, dir::Val, grid::Grid)
    indices = directional_right_boundary_indices(grid, dir)
    for i in indices
        i_left = left_index(i, dir)
        f(i_left, i)
    end
end
