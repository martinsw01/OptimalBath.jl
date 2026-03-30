export Optimizer, BFGSOptimizer, InverseSWEProblem, optimize

function no_regularization(β)
    return zero(eltype(β))
end

abstract type Optimizer end

struct BFGSOptimizer <: Optimizer end

function to_Optim(::BFGSOptimizer)
    return Optim.BFGS()
end

struct InverseSWEProblem{P, S, O, G}
    primal_problem::P
    solver_or_spec::S
    objectives::O
    gradient_type::G
    function InverseSWEProblem(primal_problem::PrimalSWEProblem,
                               primal_solver::PrimalSWESolver,
                               objectives::Objectives,
                               gradient_type::GradientType)
        return new{typeof(primal_problem),typeof(primal_solver), typeof(objectives), typeof(gradient_type)}(primal_problem, primal_solver, objectives, gradient_type)
    end
    function InverseSWEProblem(primal_problem::PrimalSWEProblem,
                               primal_solver_spec::SolverSpec,
                               objectives::Objectives,
                               gradient_type::GradientType)
        return new{typeof(primal_problem), typeof(primal_solver_spec), typeof(objectives), typeof(gradient_type)}(primal_problem, primal_solver_spec, objectives, gradient_type)
    end
end

function recording_optimizer_callback(inverse_problem::InverseSWEProblem)
    b = inverse_problem.primal_problem.initial_bathymetry
    params = ElasticMatrix(similar(b, length(b), 0))
    objectives = ElasticVector{Float64}(undef, 0)
    gradients = copy(params)

    function record(β, objective, gradient)
        push!(params, β)
        push!(objectives, objective)
        push!(gradients, gradient)
    end
    return record
end

do_nothing(β, objective, gradient) = nothing

using Optim
using NLSolversBase
function optimize(problem::InverseSWEProblem, optimizer::Optimizer, β0, callback=do_nothing)
    fg! = NLSolversBase.only_fg!() do F, G, β
        objective = compute_objective_and_gradient!(G, β, problem.solver_or_spec, problem.objectives, problem.gradient_type)
        callback(β, objective, G)
        return objective
    end
    
    opt_options = Optim.Options(iterations = 30, show_trace = true)

    res = Optim.optimize(fg!, β0, to_Optim(optimizer), opt_options)
    
    return res
end