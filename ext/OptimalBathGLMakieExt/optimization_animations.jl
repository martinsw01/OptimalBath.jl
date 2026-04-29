using ElasticArrays

function OptimalBath.animate_optimization(inverse_problem::InverseSWEProblem, β0, ::MakieBackend)
    β = ElasticMatrix(zeros(size(β0)..., 0))

    total_cost_history = Float64[]
    construction_cost_history = Float64[]
    gradient_norm_history = Float64[]

    function callback(β_new, objective, gradient)
        append!(β, β_new)
        push!(total_cost_history, objective)
        push!(construction_cost_history, inverse_problem.objectives.regularization(β_new))
        push!(gradient_norm_history, norm(gradient))
    end

    return callback, () -> animate_results(total_cost_history, construction_cost_history, gradient_norm_history, β, inverse_problem)
end

function animate_results(total_cost_history, construction_cost_history, gradient_norm_history, β, inverse_problem)
    n_lifted = Observable(1)

    f = Figure(size=(1200, 1200))
    add_bathymetry_plot!(f, β, n_lifted, inverse_problem)

    add_cost_bars!(f, total_cost_history, construction_cost_history, n_lifted, inverse_problem)

    add_gradient_norm_bar!(f, gradient_norm_history, n_lifted)

    ax = plot_results!(f[2,:], total_cost_history, construction_cost_history, gradient_norm_history, inverse_problem.objectives)
    vlines!(ax, (@lift [$n_lifted]), color=:black, linestyle=:dash)
    # @lift Slider(f[2, :], range=eachindex(total_cost_history), startvalue=$n_lifted)

    path = tempname() * "_optimization_animation.mp4"
    record(f, path, axes(β, 2); framerate = 5) do n
        n_lifted[] = n
    end
    @info "Optimization animation saved to $path"
    f
end

function add_bathymetry_plot!(f, β, n_lifted,inverse_problem)
    grid = inverse_problem.primal_problem.grid
    initial_bathymetry = inverse_problem.primal_problem.initial_bathymetry

    lifted_bathymetry = @lift initial_bathymetry .+ β[:, $n_lifted]
    lifted_title = @lift "Bathymetry optimization, iteration $($n_lifted-1)"

    ax = create_bathymetry_ax(f, lifted_title)
    
    ylims = compute_bathymetry_limits(β, initial_bathymetry, ax.xautolimitmargin[])
    x_faces = cell_faces(grid, XDIR)
    ylims!(ax, ylims...)
    plot_bathymetries!(ax, x_faces, initial_bathymetry, lifted_bathymetry)
    plot_objective_region!(ax, x_faces, inverse_problem.objectives)
    axislegend(ax, framevisible=false)
    return ax
end

function plot_bathymetries!(ax, x_faces, initial_bathymetry, modified_bathymetry)
    lines!(ax, x_faces, initial_bathymetry, label="Initial Bathymetry")
    lines!(ax, x_faces, modified_bathymetry, label="Modified Bathymetry")
end

function plot_objective_region!(ax, x_faces, objectives::Objectives)
    xstart = x_faces[objectives.objective_indices[1]]
    xend = x_faces[objectives.objective_indices[end]+1]
    vlines!(ax, [xstart, xend], color=:red, linestyle=:dash)
    vspan!(ax, xstart, xend, color=:red, alpha=0.15, label="Objective region ($(objectives.interior_objective))")
end

function add_cost_bars!(f, total_cost_history, construction_cost_history, n_lifted, inverse_problem)
    cost_ax = create_cost_ax(f)
    colsize!(f.layout, 2, Auto(0.2))

    cost_lims = compute_cost_limits(total_cost_history, construction_cost_history, cost_ax.yautolimitmargin[])
    ylims!(cost_ax, cost_lims...)

    costs_lifted = @lift [total_cost_history[$n_lifted] - construction_cost_history[$n_lifted], construction_cost_history[$n_lifted]]
    plot_cost_bars!(cost_ax, costs_lifted, cost_lims[1])
    return cost_ax
end

function plot_cost_bars!(ax, costs, min_cost)
    barplot!(ax, costs,
             color=[3, 4],
             fillto=min_cost)
end

function add_gradient_norm_bar!(f, gradient_norm_history, n_lifted)
    grad_ax = create_gradient_norm_ax(f)
    colsize!(f.layout, 3, Auto(0.1))

    grad_lims = compute_gradient_norm_limits(gradient_norm_history, grad_ax.yautolimitmargin[])
    ylims!(grad_ax, grad_lims...)

    gradient_norm_lifted = @lift [gradient_norm_history[$n_lifted]]
    plot_gradient_norm_bars!(grad_ax, gradient_norm_lifted, grad_lims[1])
    return grad_ax
end

function plot_gradient_norm_bars!(ax, gradient_norm, min_grad_norm)
    barplot!(ax, gradient_norm,
             color=[5],
             fillto=min_grad_norm)
end

function add_margin_to_lims(min_margin, max_margin, min, max)
    return min - min_margin * (max - min),
           max + max_margin * (max - min)
end

function compute_bathymetry_limits(β, initial_bathymetry, margin)
    initial_bathymetry_limits = extrema(initial_bathymetry)
    adjusted_bathymetry_limits = extrema(β) .+ initial_bathymetry_limits
    limits = minmax(initial_bathymetry_limits..., adjusted_bathymetry_limits...)
    return add_margin_to_lims(margin..., limits...)
end

function compute_cost_limits(total_cost_history, construction_cost_history, margin)
    cap_to_eps(x) = max(eps(), x)
    min_log_cost = minimum(log10 ∘ cap_to_eps, total_cost_history - construction_cost_history)
    max_log_cost = maximum(log10, total_cost_history)
    return 10 .^ add_margin_to_lims(margin..., min_log_cost, max_log_cost)
end

function compute_gradient_norm_limits(gradient_norm_history, margin)
    extrema_log_gradient_norm = extrema(log10, gradient_norm_history)
    return 10 .^ add_margin_to_lims(margin..., extrema_log_gradient_norm...)
end

create_bathymetry_ax(f, title) = Axis(
    f[1, 1],
    title=title,
    xlabel="x (m)",
    ylabel="Height (m)"
)

create_cost_ax(f) = Axis(
    f[1, 2],
    xticks=(1:2, ["Flood cost", "Construction cost"]),
    xticklabelrotation=pi/6,
    yscale=log10
)

create_gradient_norm_ax(f) = Axis(
    f[1, 3], 
    xticks=([1], ["Gradient norm"]),
    xticklabelrotation=pi/6,
    yaxisposition=:right,
    yscale=log10
)
