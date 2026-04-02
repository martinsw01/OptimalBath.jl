export State, height, momentum, States, AverageDepthStates, AverageElevationStates
export ValueType, Left, Average, Right
export HeightReference, Depth, Elevation
export to_depth!, unsafe_to_depth!, to_depth
export to_elevation!, unsafe_to_elevation!, to_elevation

using StaticArrays

abstract type ValueType end
struct Left <: ValueType end
struct Average <: ValueType end
struct Right <: ValueType end

abstract type HeightReference end
struct Depth <: HeightReference end
struct Elevation <: HeightReference end

const State = SVector

height(U::State) = U[1]
momentum(U::State, dir=1) = U[dir+1]

struct States{VT, HR, A}
    U::A
    function States{VT, HR}(U::AbstractArray{<:State}) where {VT<:ValueType, HR<:HeightReference}
        return new{VT, HR, typeof(U)}(U)
    end
end

get_dims(::Type{State{Dims, T}}) where {Dims, T} = Dims
get_float_type(::Type{A}) where A = eltype(eltype(A))

import Base: similar
function similar(U::States{VT, HR, A}, height_reference::Type{HR2}=HR, value_type::Type{VT2}=VT, float_type::Type{S}=get_float_type(A)) where {VT, HR, A, HR2, VT2, S}
    dims = get_dims(eltype(A))
    U_sim = similar(U.U, State{dims, S})
    return States{VT2, HR2}(U_sim)
end

const AverageDepthStates{A} = States{Average, Depth, A}
const AverageElevationStates{A} = States{Average, Elevation, A}

function side(b, ::Type{Left})
    return @view b[1:end-1]
end

function side(b, ::Type{Right})
    return @view b[2:end]
end

function unsafe_to_depth!(U::States{S, Elevation, A}, b) where {S, A}
    V = States{S, Depth}(U.U)
    to_depth!(V, U, b)
    return V
end

function compute_new_depth(U::State, b::T) where {T}
    h = height(U)
    p = momentum(U)
    h_depth = max(h - b, zero(T))
    # h_depth = h - b
    return State(h_depth, p)
end

function compute_new_depth(U, bl::T, br::T) where {T}
    compute_new_depth(U, 0.5 * (bl + br))
end

function compute_new_height(U, b)
    h_depth, p = U
    return State(h_depth + b, p)
end

function compute_new_height(U, bl, br)
    compute_new_height(U, 0.5 * (bl + br))
end

function to_depth!(U::States{Average, Depth, A},
                   V::States{Average, Elevation, AA},
                   b) where {A, AA}
    U.U .= compute_new_depth.(V.U, side(b, Left), side(b, Right))
end

function to_depth!(U::States{S, Depth, A},
                   V::States{S, Elevation, AA},
                   b) where {S, A, AA}
    U.U .= compute_new_depth.(V.U, side(b, S))
end

function promote_float_type(U::States, b)
    U_float_type = eltype(eltype(U.U))
    b_float_type = eltype(b)
    return promote_type(U_float_type, b_float_type)
end

function to_depth(U::States{VT, Elevation, A}, b) where {VT, A}
    U_depth = similar(U, Depth, VT, promote_float_type(U, b))
    to_depth!(U_depth, U, b)
    return U_depth
end

function to_elevation(U::States{VT, Depth, A}, b) where {VT, A}
    U_elevation = similar(U, Elevation, VT, promote_float_type(U, b))
    to_elevation!(U_elevation, U, b)
    return U_elevation
end

function to_elevation!(U::States{Average, Elevation},
                       V::States{Average, Depth},
                       b)
    U.U .= compute_new_height.(V.U, side(b, Left), side(b, Right))
end

function to_elevation!(U::States{S, Elevation},
                         V::States{S, Depth},
                         b) where S
    U.U .= compute_new_height.(V.U, side(b, S))
end

function unsafe_to_elevation!(U::States{S, Depth}, b) where S
    V = States{S, Elevation}(U.U)
    to_elevation!(V, U, b)
    return V
end

function adjust_to_bathymetry_changes!(U::States{Average, HR},
                                       V::States{Average, HR},
                                       β) where HR
    float_type = eltype(eltype(U.U))
    h = similar(U.U, float_type)
    h .= height.(V.U)
    h .+= 0.5 .* (side(β, Left) .+ side(β, Right))
    U.U .= State.(h, momentum.(V.U))
end

function adjust_to_bathymetry_changes!(U::States{Average}, β)
    iszero(β) && return
    adjust_to_bathymetry_changes!(U, U, β)
end

function adjust_to_bathymetry_changes(U::States{Average, HR},
                                      β::AbstractArray{T}) where {HR, T}
    float_type = promote_float_type(U, β)

    U_adj = similar(U, HR, Average, float_type)
    adjust_to_bathymetry_changes!(U_adj, U, β)
    return U_adj
end