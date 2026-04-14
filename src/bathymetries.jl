export b_at

const Bathymetry{Dims, T} = AbstractArray{T, Dims}

b_at(side, b::Bathymetry{dims}, j::CartesianIndex, dir=XDIR) where dims = b_at(side, b, j.I[1:dims]..., dir)

b_at(::Type{Left}, b::Bathymetry{1}, j::Int, ::XDIRT) = b[j]
b_at(::Type{Right}, b::Bathymetry{1}, j::Int, ::XDIRT) = b[j+1]

b_at(::Type{Left}, b::Bathymetry{2}, i::Int, j::Int, ::XDIRT) = 0.5 * (b[i, j] + b[i, j+1])
b_at(::Type{Right}, b::Bathymetry{2}, i::Int, j::Int, ::XDIRT) = 0.5 * (b[i+1, j] + b[i+1, j+1])
b_at(::Type{Left}, b::Bathymetry{2}, i::Int, j::Int, ::YDIRT) = 0.5 * (b[i, j] + b[i+1, j])
b_at(::Type{Right}, b::Bathymetry{2}, i::Int, j::Int, ::YDIRT) = 0.5 * (b[i, j+1] + b[i+1, j+1])

b_at(::Type{Average}, b::Bathymetry{1}, j::Int, ::Val=XDIR) = 0.5*(b[j] + b[j+1])
b_at(::Type{Average}, b::Bathymetry{2}, i::Int, j::Int, ::Val=XDIR) = 0.5 * (b_at(Left, b, i, j, XDIR) + b_at(Right, b, i, j, XDIR))
