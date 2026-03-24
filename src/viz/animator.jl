using Plots

export animate_water
include("recipes/bathymetry_recipe.jl")
include("recipes/time_recipe.jl")
include("recipes/water_recipe.jl")

# Plots.scalefontsizes(1.5)
default(;fontfamily="Computer Modern", linewidth=2)
# color_cycle = [Plots.RGB([30, 136, 229]./255...),
#                Plots.RGB([255, 193, 7]./255...),
#                Plots.RGB([216, 27, 96]./255...),]

function reshape_bathymetry(b, ylim)
    ymin, ymax = ylim
    bmin, bmax = extrema(b)

    if bmax == bmin
        return fill(ymin, size(b))
    end

    return ymin .+ (b .- bmin) .* (ymax - ymin) ./ (bmax - bmin)
end

function reshape_bathymetry(h, b, ylim)
    # Find the extrema of h and b combined
    ymin, ymax = ylim
    hmin, hmax = extrema(h)
    bmin, bmax = extrema(b)
    global_min = min(hmin, bmin)
    global_max = max(hmax, bmax)

    h_rescaled = ymin .+ (h .- global_min) .* (ymax - ymin) ./ (global_max - global_min)
    b_rescaled = ymin .+ (b .- global_min) .* (ymax - ymin) ./ (global_max - global_min)
    return h_rescaled, b_rescaled
end



function animate_adjoint_solution(Λ, t, cell_faces, b, animation_duration=t[end])
    @assert size(Λ, 2) == length(t) "Size mismatch between Λ and t."
    @assert size(Λ, 1) == length(cell_faces) - 1 "Size mismatch between Λ and cell_faces."
    @assert length(b) == length(cell_faces) "Size mismatch between b and cell_faces."

    fps, skip_frames = calc_fps(t, animation_duration)
    H = height.(Λ)
    UH = momentum.(Λ)
    HH = combine_matrices(H', H')
    UHUH = combine_matrices(UH', UH')
    x = sort([cell_faces[1:end-1]; cell_faces[2:end]])

    H_lim = calc_ylim(extrema(H)..., 0.1)
    UH_lim = calc_ylim(extrema(UH)..., 0.1)

    if isapprox(H_lim..., atol=1e-10)
        H_lim = H_lim[1] .+ (-0.1, 1.1)
    end
    if isapprox(UH_lim..., atol=1e-5)
        UH_lim = UH_lim[1] .+ (-0.1, 1.1)
    end
    
    b = reshape_bathymetry(b, H_lim .+ (0.1, -0.1))

    anim = @animate for n in eachindex(t)[end:-skip_frames:1]
        wateranim(x, HH[n, :], UHUH[n, :], "H", H_lim, UH_lim, legend=:topleft)
        adjointbathometryplot!(cell_faces, b, "Bathymetry", secondary=true)
        timeanim!(t[n], t[end], x[1], x[end], H_lim)
    end

    gif(anim, fps=fps)
end


function animate_adjoint_solution(Λ, Ul, Ur, t, cell_faces, b, animation_duration=t[end])
    @assert size(Λ, 2) == length(t) "Size mismatch between Λ and t."
    @assert size(Λ, 1) == length(cell_faces) - 1 "Size mismatch between Λ and cell_faces."
    @assert length(b) == length(cell_faces) "Size mismatch between b and cell_faces."
    # x = cell_faces
    # p = plot(x[1:end-1], height.(Ul[:, end]), label="Ul")
    # wplt = wateranim!(plot!(twinx()), x[1:end-1], 10*height.(Λ[:, end]), momentum.(Λ[:, end]), "Λ", extrema(10*height.(Λ[:, end])), extrema(momentum.(Λ[:, end])), "H", legend=:topleft)

    fps, skip_frames = calc_fps(t, animation_duration)
    λ = height.(Λ)
    μ = momentum.(Λ)
    λλ = combine_matrices(λ', λ')
    μμ = combine_matrices(μ', μ')
    x = sort([cell_faces[1:end-1]; cell_faces[2:end]])

    HH = combine_matrices(height.(Ul)', height.(Ur)')
    UHUH = abs.(combine_matrices(momentum.(Ul)', momentum.(Ur)'))

    H_lim = calc_ylim(0, maximum(HH), 0.1)
    UH_lim = calc_ylim(0, maximum(UHUH), 0.1)

    λ_lim = calc_ylim(extrema(λ)..., 0.5)
    μ_lim = calc_ylim(extrema(μ)..., 0.1)

    for lim in (H_lim, UH_lim, λ_lim, μ_lim)
        if isapprox(lim..., atol=1e-5)
            lim .= lim[1] .+ (-0.1, 1.1)
        end
    end

    anim = @animate for n in eachindex(t)[end:-skip_frames:1]
        primal_plot = plot()
        adjoint_plot = twinx(primal_plot)

        plot!(primal_plot, WaterAnim((x, HH[n, :], UHUH[n, :], "H", H_lim, UH_lim)), legend=:topleft, colorbar=false, yguide=nothing, xguide=nothing)
        plot!(primal_plot, BathometryPlot((cell_faces, b, "Bathymetry")))
        
        plot!(adjoint_plot, WaterAnim((x, λλ[n, :], μμ[n, :], "Λ", λ_lim, μ_lim)), seriescolor=:acton, legend=:topright, colorbar=false, yguide=nothing)

        # ylims!(primal_plot, H_lim)
        # ylims!(adjoint_plot, 10 .*λ_lim)
        # # adjointbathometryplot!(cell_faces, b, "Bathymetry", secondary=true)
        # timeanim!(adjoint_plot, t[n], t[end], x[1], x[end], λ_lim)
    end

    gif(anim, fps=fps)
end
    

# function animate_solution(U, t, cell_faces, b=zero(cell_faces), animation_duration=t[end])
#     return animate_solution(U, U, t, cell_faces, b, animation_duration)
# end


function animate_solution(Ul, Ur, t, cell_faces, b=zero(cell_faces), animation_duration=t[end])
    fps, skip_frames = calc_fps(t, animation_duration)
    
    hl = height.(Ul)
    hr = height.(Ur)
    pl = momentum.(Ul)
    pr = momentum.(Ur)

    HH = combine_matrices(hl', hr')
    UHUH = combine_matrices(pl', pr')
    x = sort([cell_faces[1:end-1]; cell_faces[2:end]])

    y_max = max(maximum(HH), maximum(b))
    H_lim = calc_ylim(0, y_max, 0.1)
    # UH_lim = calc_ylim(0, maximum(UHUH), 0.1)
    UH_lim = calc_ylim(extrema(UHUH)..., 0.1)
    ratio = (H_lim[2] - H_lim[1]) / (x[end] - x[1])
    @show ratio
    anim = @animate for n in eachindex(t)[1:skip_frames:end]
        wateranim(x, HH[n, :], UHUH[n, :], "H", H_lim, UH_lim, legend=:topleft, size=800 .* (1, 3ratio))
        bathometryplot!(cell_faces, b, "Bathymetry")

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
    # return anim
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