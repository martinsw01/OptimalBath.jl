using LinearAlgebra: norm
using Plots.PlotMeasures: mm

function hbar_log10(categories, values, min_value, max_value; label=nothing, kwargs...)

    min_order = log10(min_value) - 1
    max_order = log10(max_value)
    ticks_formatter = x -> "\$10^{$(round(x+min_order, digits=2))}\$"
    bar(categories, log10.(values) .- min_order,
        label=label,
        ylim=(0, max_order - min_order),
        yaxis = (formatter = ticks_formatter),
        tick_direction=:out,
        permute = (:x, :y);
        kwargs...)
end


function combine_plots(bathymetry_plots, objective_bars, gradient_norm_bars)
    plot(bathymetry_plots, objective_bars, gradient_norm_bars,
         layout=Plots.grid(3, 1, heights=(0.85, 0.1, 0.05)),
         size=(800, 500),
         right_margin=10mm)
end

function plot_objective_bars(loss, regularization, extrema_objective)
    hbar_log10(["regularization", "loss"], [regularization, loss], extrema_objective...; color=[1, 2])
end

function plot_gradient_norm(gradient_norm=1e-0, max_gradient_norm=1e-0)
    hbar_log10(["gradient norm"], [gradient_norm], 1e-8, max_gradient_norm; color=3)
end

function update_bathymetry_plot!(plt, x, b, β, n)
    if plt.n > 4
        plt.series_list[2 + n % 4].plotattributes[:y] .= b .+ β
        plt.series_list[2 + n % 4].plotattributes[:label] = "$(n)"
    else
        plot!(plt, x, b .+ β)
    end
end

function animate_optimization(inverse_problem::InverseSWEProblem, β0=zero(inverse_problem.primal_problem.initial_bathymetry))
    bathymetry_plots = plot(title="L1 regularization", xlabel="x", ylabel="Bathymetry")
    b = inverse_problem.primal_problem.initial_bathymetry
    x = range(inverse_problem.primal_problem.grid.domain..., length=length(b))
    plot!(bathymetry_plots, x, b .+ β0, label="Initial Bathymetry", linewidth=2, ylims=(-0.05, 1.5))
    n = [0]
    extrema_objective = [5e-3, 1e2]
    max_gradient_norm = [0.0]

    objective_bars = plot_objective_bars(0.0, 0.0, extrema_objective)
    gradient_norm_bars = plot_gradient_norm()

    plt = combine_plots(bathymetry_plots, objective_bars, gradient_norm_bars)

    anim = Animation()
    frame(anim, plt)
    function add_to_anim(β, objective, gradient)
        n[1] += 1
        update_bathymetry_plot!(bathymetry_plots, x, b, β, n[1])

        reg = inverse_problem.objectives.regularization(β)
        loss = objective - reg
        extrema_objective[1] = min(extrema_objective[1], objective)
        extrema_objective[2] = max(extrema_objective[2], objective)
        objective_bars = plot_objective_bars(loss, reg, extrema_objective)

        gradient_norm = norm(gradient)
        max_gradient_norm[1] = max(max_gradient_norm[1], gradient_norm)
        gradient_norm_bars = plot_gradient_norm(gradient_norm, max_gradient_norm[1])

        plt = combine_plots(bathymetry_plots, objective_bars, gradient_norm_bars)

        frame(anim, plt)
    end
    finalize_anim(args...; kwargs...) = gif(anim, args...; kwargs...)
        
    return add_to_anim, finalize_anim
end