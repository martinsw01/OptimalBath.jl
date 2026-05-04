export ADGradient

import DifferentiationInterface as DI

struct ADGradient{Preparation, ADBackend} <: GradientType
    preparation::Preparation
    ad_backend::ADBackend
    function ADGradient(ad_backend, β, spec::SolverSpec, objectives::Objectives)
        preparation = prepare_gradient(solve_and_compute_objective, ad_backend, β, spec, objectives)
        new{typeof(preparation), typeof(ad_backend)}(preparation, ad_backend)
    end
    function ADGradient(ad_backend)
        new{Nothing, typeof(ad_backend)}(nothing, ad_backend)
    end
end

function gradient!(f, g, β, spec, objectives, ad::ADGradient{Nothing})
    objective, _ = DI.value_and_gradient!(f, g, ad.ad_backend, β, DI.Constant(spec), DI.Constant(objectives))
    return objective
end

function gradient!(f, g, β, spec, objectives, ad::ADGradient)
    objective, _ = DI.value_and_gradient!(f, g, ad.preparation, ad.ad_backend, β, DI.Constant(spec), DI.Constant(objectives))
    return objective
end

function prepare_gradient(f, ad_backend, β, spec::SolverSpec, objectives::Objectives)
    return DI.prepare_gradient(f, ad_backend, β, DI.Constant(spec), DI.Constant(objectives))
end

# Automatic differentiation requires the solver to be constructed inside the AD framework, so we can only accept SolverSpecs.
function compute_objective_and_gradient!(G, β, p::PrimalSWESolver, o::Objectives, ad::ADGradient)
    @error "compute_objective_and_gradient! with ADGradient requires a SolverSpec, but got a PrimalSWESolver."
    throw(MethodError(compute_objective_and_gradient!, (G, β, p, o, ad)))
end

function compute_objective_and_gradient!(G, β, spec::SolverSpec, objectives::Objectives, ad::ADGradient)

    objective = gradient!(solve_and_compute_objective, G, β, spec, objectives, ad)

    return objective
end


include("forward_ad_gradients.jl")
include("reverse_ad_gradients.jl")
include("mooncake_gradients.jl")