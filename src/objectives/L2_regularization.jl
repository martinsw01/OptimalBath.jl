export L2

"""
    L2()

L2 regularization, divided by the number of elements in `β` to make it independent of the size of the bathymetry.

## Example
```julia-repl
julia> L2()(ones(4))
1.0
```
"""
struct L2 <: Regularization end

(::L2)(β) = sum(abs2, β) / length(β)

function add_gradient!(g, β, scale, ::L2)
    g .+= 2 .* scale .* β ./ length(β)
end