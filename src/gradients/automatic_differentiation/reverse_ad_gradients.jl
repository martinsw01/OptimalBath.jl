export ReverseADGradient, ReverseDiffBackend

import ReverseDiff

ReverseDiffBackend() = DI.AutoReverseDiff()
const ReverseADGradient = ADGradient{Nothing, DI.AutoReverseDiff{false}}
ReverseADGradient() = ADGradient(ReverseDiffBackend())


function BenchmarkReverseDiffBackend()
    @warn "Preparing the gradient with ReverseDiff should only be used for benchmarking purposes.\nThe gradient will not likely be correct, unless the parameters β are the same as those used in the preparation step,\ndefeating the purpose of preparing the gradient in the first place."
    return DI.AutoReverseDiff(compile=Val(true))
end
const BenchmarkReverseADGradient{Preparation} = ADGradient{Preparation, DI.AutoReverseDiff{true}}
BenchmarkReverseADGradient(β, spec, objectives) = ADGradient(BenchmarkReverseDiffBackend(), β, spec, objectives)


struct ReverseDiffPreparation{FunctionType, DIPreparation}
    f::FunctionType
    di_preparation::DIPreparation
end

function gradient!(f, g, β, spec, objectives, ad::BenchmarkReverseADGradient)
    objective, _ = DI.value_and_gradient!(ad.preparation.f, g, ad.preparation.di_preparation, ad.ad_backend, β)
    return objective
end

function prepare_gradient(f, ad_backend::DI.AutoReverseDiff{true}, β, spec::SolverSpec, objectives::Objectives)
    # DI will not record the wengert list if it is given context such as DI.Constant(spec) and DI.Constant(objectives).
    # We must therefore create a closure that captures the context and prepare the gradient with that closure.
    f_closure(β) = f(β, spec, objectives)
    prep = DI.prepare_gradient(f_closure, ad_backend, β)
    return ReverseDiffPreparation(f_closure, prep)
end