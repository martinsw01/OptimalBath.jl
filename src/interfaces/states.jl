export State, height, momentum, States, ValueType, Left, Average, Right, HeightReference, Depth, Elevation, CellCenters, CellFaces, to_depth!, unsafe_to_depth!, to_depth

using StaticArrays

abstract type ValueType end
struct Left <: ValueType end
struct Average <: ValueType end
struct Right <: ValueType end

abstract type HeightReference end
struct Depth <: HeightReference end
struct Elevation <: HeightReference end

const State{T} = SVector{2, T}

height(U::State) = U[1]
momentum(U::State) = U[2]

struct States{VT<:ValueType, HR<:HeightReference, T, N, A<:AbstractArray{State{T}, N}} # <: AbstractArray{State{T}, N}
    U::A
    function States{VT, HR}(U::AbstractArray{State{T}, N}) where {VT<:ValueType, HR<:HeightReference, T, N}
        return new{VT, HR, T, N, typeof(U)}(U)
    end
end

# Base.size(U::States) = size(U.U)
# Base.getindex(U::States{VT, HR, T, N, A}, I...) where {VT, HR, T, N, A} = States{VT, HR}(U.U[I...])
# Base.getindex(U::States, i::Int...) = U.U[i...]
# Base.view(U::States{VT, HR, T, N, A}, I...) where {VT, HR, T, N, A} = States{VT, HR}(view(U.U, I...))
# Base.setindex!(U::States, v, I...) = (U.U[I...] = v)
# Base.IndexStyle(::Type{<:States{VT, HR, T, N, A}}) where {VT, HR, T, N, A} = Base.IndexStyle(A)

abstract type Grid end
struct CellCenters{T, A<:AbstractArray{T}} <: Grid
    x::A
end
struct CellFaces{T, A<:AbstractArray{T}} <: Grid
    x::A
end

function side(b, ::Type{Left})
    return b[1:end-1]
end

function side(b, ::Type{Right})
    return b[2:end]
end

function unsafe_to_depth!(U::States{S, Elevation, T, N, A}, b) where {S, T, N, A}
    V = States{S, Depth}(U.U)
    to_depth!(V, U, b)
    return V
end

function to_depth!(U::States{Average, Depth, T, N, A},
                   V::States{Average, Elevation, TT, N, AA},
                   b) where {T, N, A, TT, AA}
    U.U .= State.(height.(V.U) .- 0.5 * (side(b, Left) .+ side(b, Right)), momentum.(V.U))
end

function to_depth!(U::States{S, Depth, T, N, A},
                   V::States{S, Elevation, TT, N, AA},
                   b) where {S, T, N, A, TT, AA}
    U.U .= State.(height.(V.U) .- side(b, S), momentum.(V.U))
end

function to_depth(U::States{VT, Elevation, T, N, A}, b::AbstractArray{TT}) where {VT, T, N, A, TT}
    U_depth = States{VT, Depth}(similar(U.U, State{promote_type(T, TT)}))
    to_depth!(U_depth, U, b)
    return U_depth
end