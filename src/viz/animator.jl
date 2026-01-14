export animate_water
include("recipes/bathymetry_recipe.jl")
include("recipes/time_recipe.jl")
include("recipes/water_recipe.jl")

# Plots.scalefontsizes(1.5)
default(;fontfamily="Computer Modern", linewidth=2)
# color_cycle = [Plots.RGB([30, 136, 229]./255...),
#                Plots.RGB([255, 193, 7]./255...),
#                Plots.RGB([216, 27, 96]./255...),]


function animate_solution(U, t, cell_faces, animation_duration=t[end])
    fps, skip_frames = calc_fps(t, animation_duration)
    H = getindex.(U, 1)
    UH = getindex.(U, 2)
    HH = combine_matrices(H', H')
    UHUH = combine_matrices(UH', UH')
    x = sort([cell_faces[1:end-1]; cell_faces[2:end]])

    H_lim = calc_ylim(extrema(H)..., 0.1)
    UH_lim = calc_ylim(extrema(UH)..., 0.1)
    anim = @animate for n in eachindex(t)[1:skip_frames:end]
        wateranim(x, HH[n, :], UHUH[n, :], "H", H_lim, UH_lim)
        bathometryplot!(cell_faces, zero(cell_faces), "Bathymetry")

        timeanim!(t[n], t[end], x[1], x[end], H_lim)
    end
    # @show size(x)
    # @show size(t)
    # @show size(HH)
    # for n in eachindex(t)[1:skip_frames:end]
    #     wateranim(x, HH[n, :], UHUH[n, :], "H", H_lim, UH_lim)
    #     bathometryplot!(cell_faces, zero(cell_faces), "Bathymetry")

    #     timeanim!(t[n], t[end], x[1], x[end], H_lim)
    # end

    gif(anim, fps=fps)
end


function combine_matrices(U, V)
    M, N = size(U)
    combined = Matrix{eltype(U)}(undef, M, 2N)
    combined[:, 1:2:end] .= U
    combined[:, 2:2:end] .= V
    return combined
end


function calc_ylim(min, max, padding)
    return min - padding * (max - min),
           max + padding * (max - min)
end

function calc_fps(frames, duration)
    fps_real_time = length(frames) / duration

    if fps_real_time > 50 # Max supported fps is 50 on most brwosers
        skip_frames = Int64(fps_real_time ÷ 50)
        return 50, skip_frames
    else
        return fps_real_time, 1
    end
end