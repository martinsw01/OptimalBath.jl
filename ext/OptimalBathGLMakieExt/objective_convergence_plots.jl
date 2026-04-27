function OptimalBath.plot_objective_and_gradient_norm(inverse_problem::InverseSWEProblem, ::MakieBackend)
    objective_history = Float64[]
    regularization_history = Float64[]
    gradient_norm_history = Float64[]

    function callback(β, objective, gradient)
        push!(objective_history, objective)
        push!(regularization_history, inverse_problem.objectives.regularization(β))
        push!(gradient_norm_history, norm(gradient))
    end

    return callback, (;kwargs...) -> plot_results(objective_history, regularization_history, gradient_norm_history, inverse_problem.objectives; kwargs...)
end


using LinearAlgebra: norm

function plot_results!(grid_position, objective_history, regularization_history, gradient_norm_history, objectives; scale=log10)
    # ax = Axis(grid_position, xlabel="Iteration", ylabel="Objective", yscale=scale)
    ax = Axis(grid_position, xlabel="Iteration", ylabel="Objective ($(objectives.interior_objective) + $(objectives.regularization))", yscale=scale)
    gradient_ax = Axis(grid_position, ylabel="Gradient Norm", yaxisposition=:right, yscale=scale, ygridstyle=:dot)

    plot_total_cost!(ax, objective_history)
    plot_construction_cost!(ax, regularization_history, objectives.regularization)
    plot_flood_cost!(ax, objective_history, regularization_history, objectives.interior_objective)
    add_dummy_gradient_norm_legend_entry!(ax)
    plot_gradient_norm!(gradient_ax, gradient_norm_history)
    axislegend(ax, position=:rc)

    return ax
end

function plot_results(objective_history, regularization_history, gradient_norm_history, objectives; scale=log10)
    f = Figure()
    plot_results!(f[1, 1], objective_history, regularization_history, gradient_norm_history, objectives; scale=scale)
    return f
end

function plot_total_cost!(ax, objective_history)
    scatterlines!(ax, objective_history,
           color=Cycled(1),
           label="Total cost")
end

function plot_construction_cost!(ax, regularization_history, regularization)
    scatterlines!(ax, regularization_history,
           color=Cycled(2),
           label="Construction cost ($regularization)")
end

function plot_flood_cost!(ax, objective_history, regularization_history, objective)
    scatterlines!(ax, objective_history - regularization_history,
           color=Cycled(3),
           label="Flood cost ($objective)")
end

function add_dummy_gradient_norm_legend_entry!(ax)
    scatterlines!(ax, [NaN], [NaN], color=Cycled(4), linestyle=:dash, label="Gradient norm")
end

function plot_gradient_norm!(ax, gradient_norm_history)
    scatterlines!(ax, gradient_norm_history, color=Cycled(4), linestyle=:dash)
end