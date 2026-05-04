export DIGradient
export ForwardDiffBackend, ReverseDiffBackend, MooncakeBackend, EnzymeBackend

import DifferentiationInterface as DI

const ForwardDiffBackend() = DI.AutoForwardDiff()
const ReverseDiffBackend(; compile=false) = DI.AutoReverseDiff(compile=compile)
const MooncakeBackend() = DI.AutoMooncake()


struct DIGradient{Preparation, ADBackend} <: ADGradient
    preparation::Preparation
    ad_backend::ADBackend
    function DIGradient(ad_backend, β, spec::SolverSpec, objectives::Objectives)
        preparation = prepare_gradient(solve_and_compute_objective, ad_backend, β, spec, objectives)
        new{typeof(preparation), typeof(ad_backend)}(preparation, ad_backend)
    end
    function DIGradient(ad_backend)
        new{Nothing, typeof(ad_backend)}(nothing, ad_backend)
    end
end

function gradient!(f, g, β, spec, objectives, ad::DIGradient{Nothing})
    objective, _ = DI.value_and_gradient!(f, g, ad.ad_backend, β, DI.Constant(spec), DI.Constant(objectives))
    return objective
end

function gradient!(f, g, β, spec, objectives, ad::DIGradient)
    objective, _ = DI.value_and_gradient!(f, g, ad.preparation, ad.ad_backend, β, DI.Constant(spec), DI.Constant(objectives))
    return objective
end

function prepare_gradient(f, ad_backend, β, spec::SolverSpec, objectives::Objectives)
    return DI.prepare_gradient(f, ad_backend, β, DI.Constant(spec), DI.Constant(objectives))
end




struct ReverseDiffPreparation{FunctionType, DIPreparation}
    f::FunctionType
    di_preparation::DIPreparation
end

function gradient!(f, g, β, spec, objectives, ad::DIGradient{<:ReverseDiffPreparation})
    objective, _ = DI.value_and_gradient!(ad.preparation.f, g, ad.preparation.di_preparation, ad.ad_backend, β)
    return objective
end

function prepare_gradient(f, ad_backend::DI.AutoReverseDiff, β, spec::SolverSpec, objectives::Objectives)
    @assert ad_backend.compile "ReverseDiff preparation with compile=false makes preparation obsolete. Use ReverseDiffBackend(compile=true) instead."
    @warn "Preparing the gradient with ReverseDiff should only be used for benchmarking purposes.\nThe gradient will now unlikely be correct, unless the parameters β are the same as those used in the preparation step, defeating the purpose of preparing the gradient in the first place."
    # DI will not record the wengert list if it is given context such as DI.Constant(spec) and DI.Constant(objectives).
    # We must therefore create a closure that captures the context and prepare the gradient with that closure.
    f_closure(β) = f(β, spec, objectives)
    prep = DI.prepare_gradient(f_closure, ad_backend, β)
    return ReverseDiffPreparation(f_closure, prep)
end
