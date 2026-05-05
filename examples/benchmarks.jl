using OptimalBath
using GLMakie
using DataFrames

function prepare_gradient(N, ad_type::Type{<:ADGradient}, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    objectives = create_objectives(problem.grid)
    gradient_type = ad_type(β, spec, objectives)
    return gradient_type, β, spec, objectives
end

function prepare_gradient(N, ad_type::Type{<:ReverseADGradient}, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    objectives = create_objectives(problem.grid)
    gradient_type = ad_type()
    return gradient_type, β, spec, objectives
end

function prepare_gradient(N, ad::Type{DiscreteAdjointGradient}, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    objectives = create_objectives(problem.grid)
    gradient_type = ad(solver)
    return gradient_type, β, solver, objectives
end

function compare_gradient_types(resolutions, gradient_types)
    @info "Warming up"
    for gradient_type in gradient_types
        println("        - $gradient_type")
        grad_type, β, solver_or_spec, objectives = prepare_gradient(resolutions[1], gradient_type)
        compute_objective_and_gradient(β, solver_or_spec, objectives, grad_type)
    end

    function time_gradient_computation(N, gradient_type_or_backend)
        @info "Preparing gradient, resolution=$N, gradient_type=$gradient_type_or_backend"
        prep_res = @timed gradient_type, β, solver_or_spec, objectives = prepare_gradient(N, gradient_type_or_backend)
        @show prep_res.time
        @info "Computing gradient, resolution=$N, gradient_type=$gradient_type_or_backend"
        res = @timed compute_objective_and_gradient(β, solver_or_spec, objectives, gradient_type)
        @show res.time
        return res
    end

    results = DataFrame(
        gradient_type = Type{<:GradientType}[],
        resolutions = Tuple{Int, Int}[],
        time = Float64[],
        bytes = Int[],
        gctime = Float64[]
    )

    for gradient_type in gradient_types
        for N in resolutions
            res = time_gradient_computation(N, gradient_type)
            push!(results, (gradient_type, N, res.time, res.bytes, res.gctime))
        end
    end

    return results
end

function plot_comparison(results::DataFrame)
    f = Figure(title="Gradient computation time comparison")
    ax = Axis(f[1, 1], title="Gradient computation time comparison", xlabel="Resolution (Nx * Ny)", ylabel="Time (s)", yscale=log10, xscale=log10)
    # ax2 = Axis(f[1, 1], yaxisposition=:right, ylabel="Time fraction", yscale=log10, xscale=log10, ygridstyle=:dash)

    for df in groupby(results, :gradient_type)
        sort!(df, :resolutions)
        gradient_type = first(df.gradient_type)
        scatterlines!(ax, prod.(df.resolutions), df.time, label=string(gradient_type))
    end
        


    # scatterlines!(ax, prod.(resolutions), adjoint_times, label="Discrete adjoint")
    # scatterlines!(ax, prod.(resolutions), forward_ad_times, label="Forward AD")
    # scatterlines!(ax2, prod.(resolutions), forward_ad_times ./ adjoint_times, linestyle=:dash)
    # scatterlines!(ax, [NaN], [NaN], color=Cycled(1), label="FAD / DA", linestyle=:dash)
    axislegend(ax, position=:lt)
    return f
end