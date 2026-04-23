function OptimalBath.plot_gradient(g, b, U0, grid::Grid{1}, backend::MakieBackend)
    f, ax = create_water_figure_and_axis()
    _plot_water_and_gradient!(f, ax, g, b, U0, grid, backend)
    axislegend(ax)
    return f
end

function OptimalBath.plot_gradient(g, b, U0, objectives::Objectives, grid::Grid{1}, backend::MakieBackend)
    f, ax = create_water_figure_and_axis()
    _plot_water_and_gradient!(f, ax, g, b, U0, grid, backend)
    plot_objective_region!(ax, objectives, grid)
    axislegend(ax)
    return f
end

function _plot_water_and_gradient!(f, ax, g, b, U0, grid::Grid{1}, backend::MakieBackend)
    _plot_water!(f, ax, b, U0, U0, grid, backend; colorbar=false)
    _plot_gradient!(f, ax, g, grid, backend)
    add_dummy_gradient_legend_entry!(ax)
    return f, ax
end

function add_dummy_gradient_legend_entry!(ax)
    lines!(ax, [NaN], [NaN], color=:black, label="Gradient")
end

function _plot_gradient!(f, ax, g, grid::Grid{1}, ::MakieBackend)
    grad_ax = Axis(f[1, 1], yaxisposition = :right, ylabel="Gradient")
    lines!(grad_ax, cell_faces(grid, XDIR), g, color=:black)
    hlines!(grad_ax, [0], linestyle=:dash, color=:black)
    return grad_ax
end

function plot_objective_region!(ax, objectives::Objectives, grid::Grid{1})
    xstart = cell_faces(grid, XDIR)[objectives.objective_indices[1]]
    xend = cell_faces(grid, XDIR)[objectives.objective_indices[end] + 1]
    vlines!(ax, [xstart, xend], linestyle=:dash, color=:red, label=nothing)
    vspan!(ax, xstart, xend, color=:red, alpha=0.15, label="Objective region ($(objectives.interior_objective))")
end