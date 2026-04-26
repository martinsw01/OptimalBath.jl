function OptimalBath.plot_gradient(g, b, grid::Grid{2}, ::MakieBackend)
    f = Figure()
    ax = plot_bathymetry_and_gradient!(f, b, g, grid)
    axislegend(ax)
    return f
end

function OptimalBath.plot_gradient(g, b, objective::Objectives, grid::Grid{2}, ::MakieBackend)
    f = Figure()
    ax = plot_bathymetry_and_gradient!(f, b, g, grid)
    plot_objective_region!(ax, objective, grid)
    axislegend(ax)
    return f
end


create_2d_gradient_axis(f) = Axis(
    f[1, 1],
    title="Bathymetry and gradient",
    xlabel="x",
    ylabel="y"
)

function compute_main_contour_levels(b, levels)
    # Use makie to compute "nice" readable levels
    Makie.get_tickvalues(LinearTicks(levels), extrema(b)...)
end
function compute_secondary_contour_levels(main_levels)
    main_Δ = main_levels[2] - main_levels[1]
    return main_levels[1]-main_Δ:main_Δ/5:main_levels[end]+main_Δ
end

function plot_bathymetry_contours!(ax, n_main_levels, b, grid::Grid{2})
    main_contour_levels = compute_main_contour_levels(b, n_main_levels)
    secondary_contour_levels = compute_secondary_contour_levels(main_contour_levels)

    contour!(ax, cell_faces(grid)..., b,
             color=(:black, 0.2),
             levels=secondary_contour_levels)
    contour!(ax, cell_faces(grid)..., b,
             color=:black,
             labels=true,
             levels=main_contour_levels)

    add_dummy_bathymetry_legend_entry!(ax)
end


function plot_gradient!(ax, g, grid::Grid{2})
    gmax = maximum(abs, g)
    levels = range(-gmax, gmax, 12)
    contourf!(ax, cell_faces(grid)..., g,
             colormap=:balance,
             levels=levels)
end

function plot_colorbar!(f, gradient_plot)
    Colorbar(f[1, 2], gradient_plot, label="Gradient")
end

function add_dummy_bathymetry_legend_entry!(ax)
    lines!(ax, [NaN], [NaN], label="Bathymetry contours", color=:black)
end

function plot_bathymetry_and_gradient!(f, b, g, grid::Grid{2})
    ax = create_2d_gradient_axis(f)

    g_contourf = plot_gradient!(ax, g, grid)
    plot_colorbar!(f, g_contourf)
    plot_bathymetry_contours!(ax, 6, b, grid)

    return ax
end

function plot_objective_region!(ax, objectives::Objectives, grid::Grid{2})
    x_faces = cell_faces(grid, XDIR)
    y_faces = cell_faces(grid, YDIR)

    indices = objectives.objective_indices
    x_min = x_faces[indices[1, 1][1]]
    x_max = x_faces[indices[end, 1][1] + 1]
    y_min = y_faces[indices[end, 1][2]]
    y_max = y_faces[indices[1, end][2] + 1]

    lines!(ax, [x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
           color=:red,
           linewidth=2,
           label="Objective region")
end