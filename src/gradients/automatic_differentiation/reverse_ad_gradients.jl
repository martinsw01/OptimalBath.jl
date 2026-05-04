export ReverseADGradient, ReverseDiffBackend

import ReverseDiff

const ReverseDiffBackend(; compile=Val(false)) = DI.AutoReverseDiff(compile=compile)
const ReverseADGradient{Preparation} = ADGradient{Preparation, <:DI.AutoReverseDiff}
const ReverseADGradient(args...; compile=Val(false)) = ADGradient(ReverseDiffBackend(compile=compile), args...)

struct ReverseDiffPreparation{FunctionType, DIPreparation}
    f::FunctionType
    di_preparation::DIPreparation
end

function gradient!(f, g, β, spec, objectives, ad::ADGradient{<:ReverseDiffPreparation})
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