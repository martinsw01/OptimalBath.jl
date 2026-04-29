function OptimalBath.animate_solution(Ul, Ur, t, bathymetry, grid::Grid{1}, animation_duration, ::MakieBackend)
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

    path = tempname() * ".mp4"
    framerate = Int(length(t) ÷ animation_duration)
    record(f, path, eachindex(t), framerate=framerate) do n
        n_lifted[] = n
    end
    @info "Animation saved to $path"

    return f
end

function OptimalBath.animate_solution(Ul, Ur, t, bathymetry, grid::Grid{1}, backend::MakieBackend)
    return animate_solution(Ul, Ur, t, bathymetry, grid, t[end], backend)
end