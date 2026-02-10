export State, height, momentum, States, ValueType, Left, Average, Right, HeightReference, Depth, Elevation, to_depth!, unsafe_to_depth!, to_depth

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

function side(b, ::Type{Left})
    return @view b[1:end-1]
end

function side(b, ::Type{Right})
    return @view b[2:end]
end

function unsafe_to_depth!(U::States{S, Elevation, T, N, A}, b) where {S, T, N, A}
    V = States{S, Depth}(U.U)
    to_depth!(V, U, b)
    return V
end

function compute_new_depth(U::State, b::T) where {T}
    h = height(U)
    p = momentum(U)
    h_depth = max(h - b, zero(T))
    # h_depth = h - b
    return State{T}(h_depth, p)
end

function compute_new_depth(U, bl::T, br::T) where {T}
    compute_new_depth(U, 0.5 * (bl + br))
end

function to_depth!(U::States{Average, Depth, T, N, A},
                   V::States{Average, Elevation, TT, N, AA},
                   b::AbstractArray{T}) where {T, N, A, TT, AA}
    U.U .= compute_new_depth.(V.U, side(b, Left), side(b, Right))
end

function to_depth!(U::States{S, Depth, T, N, A},
                   V::States{S, Elevation, TT, N, AA},
                   b::AbstractArray{T}) where {S, T, N, A, TT, AA}
    U.U .= compute_new_depth.(V.U, side(b, S))
end

function to_depth(U::States{VT, Elevation, T, N, A}, b::AbstractArray{TT}) where {VT, T, N, A, TT}
    U_depth = States{VT, Depth}(similar(U.U, State{promote_type(T, TT)}))
    to_depth!(U_depth, U, b)
    return U_depth
end

function adjust_to_bathymetry_changes!(U::States{Average, HR, T, N, A},
                                       V::States{Average, HR, TT, N, AA},
                                       β::AbstractArray{T}) where {HR, T, N, A, TT, AA}
    h = similar(U.U, T)
    h .= height.(V.U)
    h .+= 0.5 .* (side(β, Left) .+ side(β, Right))
    U.U .= State.(h, momentum.(V.U))
end

function adjust_to_bathymetry_changes!(U::States{Average, HR, T, N, A},
                                       β::AbstractArray{T}) where {HR, T, N, A}
    iszero(β) && return
    adjust_to_bathymetry_changes!(U, U, β)
end

function adjust_to_bathymetry_changes(U::States{Average, HR, T, N, A},
                                      β::AbstractArray{TT}) where {HR, T, N, A, TT}
    T_promoted = State{promote_type(T, TT)}
    U_adj = States{Average, HR}(similar(U.U, T_promoted))
    adjust_to_bathymetry_changes!(U_adj, U, β)
    return U_adj
end