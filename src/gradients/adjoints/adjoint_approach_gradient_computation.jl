function _adjoint_based_gradient!(G, Λ, U, t, design_indices::Colon, grid::Grid{1}, ca::AdjointApproachGradient)
    _adjoint_based_gradient!(G, Λ, U, t, eachindex(G), grid, ca)
end

function _adjoint_based_gradient!(G, Λ, U, t, design_indices::AbstractVector{Int}, grid::Grid{1}, ca::AdjointApproachGradient)
    N, M = size(U)
    for (i, j) in enumerate(design_indices)
        if j == 1
            G[i] += integrate_gradient(U[j,:], Λ[j,:], t, XDIR, grid)
        elseif j == N + 1
            G[i] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, XDIR, grid)
        else
            G[i] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, XDIR, grid)
            G[i] += integrate_gradient(U[j,:], Λ[j,:], t, XDIR, grid)
        end
    end
end

function _adjoint_based_gradient!(G, Λ, U, t, design_indices::AbstractUnitRange{<:Integer}, grid::Grid{1}, ca::AdjointApproachGradient)
    start = first(design_indices)
    stop = last(design_indices) - 1
    for j in start:stop
        temp_integral = integrate_gradient(U[j,:], Λ[j,:], t, XDIR, grid)
        G[j] += temp_integral
        G[j+1] -= temp_integral
    end
end

@views function _adjoint_based_gradient!(G, Λ, U, t, design_indices::CartesianIndices, grid::Grid, ::AdjointApproachGradient)
    N_corners = size(G)
    N_cells = N_corners .- 1

    w = contribution_weight(grid)

    for dir in directions(grid)
        for (s, offset) in zip(signs(dir, grid), offsets(grid))
            for i in CartesianIndices(G)
                ui = i + offset
                if in_bounds(ui, N_cells)
                    I = design_indices[ui]
                    G[i] += w * s * integrate_gradient(U[I, :], Λ[I, :], t, dir, grid)
                end
            end
        end
    end
end

function integrate_gradient(U, Λ, t, dir, grid::Grid)
    time_indices = firstindex(t):lastindex(t)-1

    Δx = get_Δx(grid, dir)
    gradient = zero(eltype(eltype(U)))
    @inbounds @simd for n in time_indices
        Δt = t[n+1] - t[n]
        gradient += momentum(Λ[n+1], dir) * height(U[n]) * Δt
    end
    return gradient * 9.81 / Δx
end

signs(::XDIRT, ::Grid{2}) = (1.0, -1.0,  1.0, -1.0)
signs(::YDIRT, ::Grid{2}) = (1.0,  1.0, -1.0, -1.0)
offsets(::Grid{2}) = map(CartesianIndex{2}, ((0, 0), (-1, 0), (0, -1), (-1, -1)))

signs(::XDIRT, ::Grid{1}) = (1.0, -1.0)
offsets(::Grid{1}) = map(CartesianIndex{1}, (0, -1))

in_bounds(i, N) = i in CartesianIndices(N)

contribution_weight(::Grid{D}) where D = 2.0^(1-D)