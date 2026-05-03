export DifferentiationInterfaceGradient

struct DifferentiationInterfaceGradient{Preparation, ADBackend} <: ADGradient
    ad_backend::ADBackend
    preparation::Preparation
    function DifferentiationInterfaceGradient(ad_backend, β, spec::SolverSpec, objectives::Objectives)
        preparation = prepare_gradient(solve_and_compute_objective, ad_backend, β, Const(spec), Const(objectives))
        new{typeof(ad_backend), typeof(preparation)}(ad_backend, preparation)
    end
    function DifferentiationInterfaceGradient(ad_backend)
        new{typeof(ad_backend), Nothing}(ad_backend, nothing)
    end
end

function gradient!(f, g, β, spec, objectives, ad::DifferentiationInterfaceGradient{Nothing})
    objective, _ = DifferentiationInterface.value_and_gradient!(f, g, ad.ad_backend, β, Const(spec), Const(objectives))
    return objective
end

function gradient!(f, g, β, spec, objectives, ad::DifferentiationInterfaceGradient)
    objective, _ = DifferentiationInterface.value_and_gradient!(f, g, ad.preparation, ad.ad_backend, β, Const(spec), Const(objectives))
    return objective
end