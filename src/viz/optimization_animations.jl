function animate_optimization(inverse_problem::InverseSWEProblem, β0=zero(inverse_problem.primal_problem.initial_bathymetry))
    plt = plot()
    b = inverse_problem.primal_problem.initial_bathymetry
    x = range(inverse_problem.primal_problem.domain..., length=length(b))
    plot!(plt, x, b .+ β0, label="Initial Bathymetry", linewidth=2)
    n = [0]

    anim = Animation()
    frame(anim, plt)
    function add_to_anim(β, objective, gradient)
        n[1] += 1
        if plt.n > 4
            plt.series_list[2 + n[1] % 4].plotattributes[:y] .= b .+ β
            plt.series_list[2 + n[1] % 4].plotattributes[:label] = "$(n[1])"
        else
            plot!(plt, x, b .+β)
        end
        frame(anim, plt)
    end
    finalize_anim(args...; kwargs...) = gif(anim, args..., kwargs...)
        
    return add_to_anim, finalize_anim
end