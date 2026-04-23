using OptimalBath: height

function plot_water(b, Ul, Ur, grid::Grid{1}, ::MakieBackend)
    f, ax = create_water_figure_and_axis()
    _plot_water!(f, ax, b, Ul, Ur, grid, MakieBackend())
    axislegend(ax)
    return f
end

function create_water_figure_and_axis()
    f = Figure()
    ax = Axis(f[1, 1], xlabel="x", ylabel="Height above reference (m)")
    return f, ax
end

function _plot_water!(f, ax, b, Ul, Ur, grid::Grid{1}, ::MakieBackend; colorbar=true)
    hlim = compute_hlims(Ul, Ur, 0.1)
    plim = compute_plims(Ul, Ur, 0.1)

    hh = combine_vectors(height, Ul, Ur)
    pp = combine_vectors(abs ∘ momentum, Ul, Ur)

    xx = sort([cell_faces(grid, XDIR)[1:end-1]; cell_faces(grid, XDIR)[2:end]])

    plot_water(f, ax, hh, pp, xx, hlim, plim, grid; colorbar=colorbar)
    plot_bathymetry(ax, b, grid)

    return f, ax
end

function plot_bathymetry(ax, bathymetry, grid::Grid{1})
    x_faces = cell_faces(grid, XDIR)
    band!(ax, x_faces, 0, bathymetry, color=:brown, label="Bathymetry")
end

function fill_left_and_right!(f, aa, left, right)
    aa[1:2:end] .= f.(left)
    aa[2:2:end] .= f.(right)
end

function add_dummy_legend_entry!(ax, label, color)
    scatter!(ax, [NaN], [NaN], color=color, label=label, marker=:rect, markersize=31)
end

function combine_vectors(f, left, right)
    combined = similar(left, typeof(f(left[1])), length(left) + length(right))
    combined[1:2:end] .= f.(left)
    combined[2:2:end] .= f.(right)
    return combined
end

function plot_water(f, ax, hh, pp, xx, hlim, plim, grid::Grid{1}; colorbar=true)
    # add clims = plim to the band!
    pl = band!(ax, xx, 0, hh, color=pp, colormap=:blues, colorrange=plim)
    add_dummy_legend_entry!(ax, "Water", :lightblue)
    if colorbar
        Colorbar(f[1, 2], pl, label="Magnitude of momentum")
    end
    ylims!(ax, hlim)
end

function compute_time_position(tn, t, grid::Grid{1})
    x_L, x_R = extrema(cell_faces(grid, XDIR))
    t_end = last(t)
    return (x_R - x_L) * tn / t_end + x_L
end

function plot_time(ax, y, x_t, t, grid::Grid{1})
    x_L, x_R = extrema(cell_faces(grid, XDIR))

    lines!(ax, [x_L, x_R], [y, y], color=:lightgray, linewidth=10)
    lines!(ax, (@lift [x_L, $x_t]), [y, y], color=:grey, linewidth=10, label="Time: $(t[]) s")
end

function Base.minmax(a, b, c, d)
    return min(a, b, c, d), max(a, b, c, d)
end

function compute_hlims(Ul, Ur, padding)
    hmin, hmax = minmax(extrema(height, Ul)..., extrema(height, Ur)...)
    return hmin - padding * (hmax - hmin),
           hmax + padding * (hmax - hmin)
end

function compute_plims(Ul, Ur, padding)
    pmax = max(maximum(abs ∘ momentum, Ul), maximum(abs ∘ momentum, Ur))
    return (zero(pmax), pmax + padding * pmax)
end