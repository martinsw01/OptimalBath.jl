export Regularization, NoRegularization
export gradient!, add_gradient!

"""
    Regularization
Defines the building cost of modifying the bathymetry.
It is defined as vector space of regularization terms,
so that we can easily make linear combinations of regularization terms.

## Example
```
julia> regularization = 0.1 * L2() + 0.5 * SoftL1(10.0)
0.1 * L2() + 0.5 * SoftL1(10.0)
julia> typeof(regularization)
RegularizationSum{ScaledRegularization{L2, Float64}, ScaledRegularization{SoftL1, Float64}}
```
"""
abstract type Regularization end

"""
    gradient!(g, β, regularization)
Computes the gradient of the regularization term with respect to the bathymetry `β` and stores it in `g`.

## Example
```julia-repl
regularization = 0.1 * L2() + 0.5 * SoftL1(10.0)
0.1 * L2() + 0.5 * SoftL1(10.0)
julia> β = 1.0:4.0;
julia> g = similar(β);
julia> gradient!(g, β, regularization)
4-element Vector{Float64}:
 0.1749886505328244
 0.2249999994847116
 0.2749999999999766
 0.325
```
"""
function gradient!(g, β, regularization::Regularization)
    fill!(g, zero(eltype(β)))
    add_gradient!(g, β, 1, regularization)
end

struct ScaledRegularization{RegType, FloatType} <: Regularization
    scale::FloatType
    regularization::RegType
end

(r::ScaledRegularization)(β) = r.scale * r.regularization(β)

function Base.:*(scale::Number, regularization::Regularization)
    return ScaledRegularization(scale, regularization)
end

function Base.:/(regularization::Regularization, scale::Number)
    return ScaledRegularization(1/scale, regularization)
end

function Base.show(io::IO, regularization::ScaledRegularization)
    return print(io, "$(regularization.scale) * $(repr(regularization.regularization))")
end

function add_gradient!(g, β, scale, r::ScaledRegularization)
    add_gradient!(g, β, scale * r.scale, r.regularization)
end

struct RegularizationSum{RegType1, RegType2} <: Regularization
    r1::RegType1
    r2::RegType2
end

(r::RegularizationSum)(β) = r.r1(β) + r.r2(β)

function Base.:+(r1::Regularization, r2::Regularization)
    return RegularizationSum(r1, r2)
end

function Base.:-(r1::Regularization, r2::Regularization)
    return RegularizationSum(r1, -1 * r2)
end

function Base.show(io::IO, regularization::RegularizationSum)
    return print(io, "$(repr(regularization.r1)) + $(repr(regularization.r2))")
end

function add_gradient!(g, β, scale, r::RegularizationSum)
    add_gradient!(g, β, scale, r.r1)
    add_gradient!(g, β, scale, r.r2)
end

"""
    NoRegularization()
"""
struct NoRegularization <: Regularization end

Base.:*(::Number, no_regularization::NoRegularization) = no_regularization

(::NoRegularization)(β) = zero(eltype(β))

add_gradient!(g, β, scale, ::NoRegularization) = nothing


function add_gradient!(g, β, scale, regularization)
    tmp = similar(g)
    ReverseDiff.gradient!(tmp, regularization, β)
    @. g += scale * tmp
end

include("L2_regularization.jl")
include("soft_L1_regularization.jl")
include("soft_TV_regularization.jl")