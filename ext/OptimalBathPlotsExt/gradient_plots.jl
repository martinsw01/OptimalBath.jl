function OptimalBath.plot_gradient(g, b, U0, grid::Grid{1}, ::PlotsBackend)
    x = cell_faces(grid, XDIR)
    x = sort([x[1:end-1]; x[2:end]])

    H = height.(U0)
    H = max.(H, 0.5 * (b[1:end-1] + b[2:end]))
    P = momentum.(U0)
    HH = combine_matrices(H', H')[1,:]
    PP = combine_matrices(P', P')[1,:]
    H_lim = calc_ylim(extrema(H)..., 0.1)

    background_plot = plot()
    gradient_plot = twinx(background_plot)

    plot!(background_plot, WaterAnim((x, HH, PP, "Initial condition", H_lim, (0, 1))), colorbar=false, legend=:topright, ylabel="Height (m)")
    plot!(background_plot, BathometryPlot((cell_faces(grid, XDIR), b, "Bathymetry")))#, seriescolor="#aa5c27")
    plot!(background_plot, [0.], [NaN], label="Gradient", color="#000000", linewidth=2)

    hline!(gradient_plot, [0], linestyle=:dash, color=:black, label=nothing)
    plot!(gradient_plot, cell_faces(grid, XDIR), g, label=nothing, ylabel="∂J/∂β", color="#000000", linewidth=2)
    
    background_plot[1][:yaxis][:mirror] = true
    background_plot[2][:yaxis][:mirror] = false
    return background_plot
end

function OptimalBath.plot_gradient(g, b, U, objectives::Objectives, grid::Grid{1}, backend::PlotsBackend)
    plt = plot_gradient(g, b, U, grid, backend)
    plot_objective_region!(plt, objectives, grid)
    return plt
end

function plot_objective_region!(plt, objectives::Objectives, grid::Grid{1})
    xstart = cell_faces(grid, XDIR)[objectives.objective_indices[1]]
    xend = cell_faces(grid, XDIR)[objectives.objective_indices[end] + 1]
    vline!(plt, [xstart, xend], linestyle=:dash, color=:red, label=nothing)
    vspan!(plt, [xstart, xend],  fillcolor=:red, fillalpha=0.1, label="Objective region ($(objectives.interior_objective))")
end

function gradient_contourf!(plt, x_faces, y_faces, g)
    m = maximum(abs, g)
    m = (m == 0) ? 1.0 : m
    contourf(plt, x_faces, y_faces, g',
             xlabel="x",
             ylabel="y", 
             colorbar_title="∂J/∂β",
             clims=(-m, m),
             ylims=extrema(y_faces),
             xlims=extrema(x_faces),
             linewidth=0,
             color=:balance)
end

function bathymetry_contour!(plt, x_faces, y_faces, b, g_clims)
    bmin, bmax = extrema(b)
    levels = (bmin ≈ bmax) ? [bmin] : range(bmin, bmax; length=12)[2:end-1]

    # Map bathymetry values into gradient clims so contour lines are always drawable.
    scale_to_gradient(v) = g_clims[1] + (v - bmin) * (g_clims[2] - g_clims[1]) / (bmax - bmin)
    b_scaled = if bmin ≈ bmax
        fill((g_clims[1] + g_clims[2]) / 2, size(b))
    else
        scale_to_gradient.(b)
    end
    scaled_levels = (bmin ≈ bmax) ? [(g_clims[1] + g_clims[2]) / 2] : scale_to_gradient.(levels)

    contour!(plt, x_faces, y_faces, b_scaled',
             label="Bathymetry contours",
             color=fill(:black, length(scaled_levels)),
             levels=scaled_levels,
             linewidth=1.5,
             colorbar_entry=false,
             clabels=false)

    # Show original-unit bathymetry labels near actual contour locations.
    for lvl in levels
        # Find a grid point where bathymetry is closest to the current contour level.
        idx = argmin(abs.(b .- lvl))
        i, j = Tuple(CartesianIndices(b)[idx])
        x = x_faces[i]
        y = max(y_faces[j], 10)
        annotate!(plt, x, y, text(string(round(lvl, sigdigits=3)), 8, :black))
    end
    plot!(plt, [], [], color=:black, label="Bathymetry contours")
end

function OptimalBath.plot_gradient(g, b, grid::Grid{2}, ::PlotsBackend)
    x_faces = cell_faces(grid, XDIR)
    y_faces = cell_faces(grid, YDIR)
    m = maximum(abs, g)
    m = (m == 0) ? 1.0 : m
    plt = plot()
    plt = gradient_contourf!(plt, x_faces, y_faces, g)
    bathymetry_contour!(plt, x_faces, y_faces, b, (-m, m))
    return plt
end

function plot_objective!(plt, objectives::Objectives, grid::Grid{2})
    x_faces = cell_faces(grid, XDIR)
    y_faces = cell_faces(grid, YDIR)

    indices = objectives.objective_indices
    x_min = x_faces[indices[1, 1][1]]
    x_max = x_faces[indices[end, 1][1] + 1]
    y_min = y_faces[indices[end, 1][2]]
    y_max = y_faces[indices[1, end][2] + 1]

    plot!(plt, [x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
          color=:red,
          linewidth=2,
          label="Objective region")
end

function OptimalBath.plot_gradient(g, b, objectives::Objectives, grid::Grid{2}, backend::PlotsBackend)
    plt = plot_gradient(g, b, grid, backend)
    plot_objective!(plt, objectives, grid)
    return plt
end