function OptimalBath.animate_solution(Ul, Ur, t, bathymetry, grid::Grid{1}, ::MakieBackend; duration=t[end])
    n_lifted = Observable(firstindex(t))

    f, ax = create_water_figure_and_axis()
    
    hlim = compute_hlims(Ul, Ur, 0.1)
    plim = compute_plims(Ul, Ur, 0.1)

    h_lifted = @lift combine_vectors(height, Ul[:, $n_lifted], Ur[:, $n_lifted])
    p_lifted = @lift combine_vectors(abs ∘ momentum, Ul[:, $n_lifted], Ur[:, $n_lifted])

    xx = sort([cell_faces(grid, XDIR)[1:end-1]; cell_faces(grid, XDIR)[2:end]])

    plot_water(f, ax, h_lifted, p_lifted, xx, hlim, plim, grid)
    plot_bathymetry(ax, bathymetry, grid)
    plot_time(f, t, n_lifted)
    axislegend(ax)

    path = tempname() * "_simulation_animation_1d.mp4"
    framerate, skip_frames = compute_capped_framerate(length(t), duration)
    record(f, path, eachindex(t)[1:skip_frames:end], framerate=framerate) do n
        n_lifted[] = n
    end
    @info "Animation saved to $path"

    return f
end

function compute_capped_framerate(frames, duration, max_fps=30)
    fps = Int(frames ÷ duration)
    capped_fps = min(fps, max_fps)
    skip_frames = Int(frames ÷ (capped_fps * duration))
    return capped_fps, skip_frames
end

function OptimalBath.animate_solution(U, t, bathymetry, grid::Grid{1}, backend::MakieBackend; duration=t[end])
    return animate_solution(U, U, t, bathymetry, grid, backend; duration=duration)
end