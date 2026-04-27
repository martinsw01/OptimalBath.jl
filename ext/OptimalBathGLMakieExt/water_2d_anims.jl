function OptimalBath.animate_solution(U, t, b, grid::Grid{2}, ::MakieBackend; duration=t[end])
    f = Figure()
    ax = create_simulation_axis(f)

    n = Observable(firstindex(t))
    water_elevation = @lift OptimalBath.height.(U[:, :, $n]) .- 0.02
    water_color = @lift depth_color(U[:, :, $n], b)

    plot_bathymetry_surface!(ax, grid, b)

    plot_water_surface!(ax, grid, water_elevation, water_color)

    plot_progress_bar(f, t, n)

    path = tempname() * "_flood_animation.mp4"
    record(f, path, eachindex(t), framerate=30) do i
        n[] = i
    end
    @info "Animation saved to: $path"

    f
end

function depth_color(U, b)
    return height.(to_depth(States{Average, Elevation}(U), b).U)
end

create_simulation_axis(f) =  Axis3(
    f[1, 1],
    xlabel="x (m)",
    ylabel="y (m)",
    zlabel="Height (m)",
    title="Flood simulation",
    azimuth=1.7pi
)

function plot_bathymetry_surface!(ax, grid, b)
    colorrange = add_margin_to_lims(0.2, 0.5, extrema(b)...)
    surface!(ax, cell_faces(grid)..., b,
             colormap=:greenbrownterrain,
             label="Bathymetry",
             colorrange=colorrange)
end

function plot_water_surface!(ax, grid, water_elevation, water_color)
    surface!(ax, cell_centers(grid)..., water_elevation,
             color = water_color,
             colormap=(:blues, 0.6),
             label="Water height")
end

function plot_progress_bar(f, t, n)
    @lift Slider(f[2, 1], range=eachindex(t), startvalue=$n)
end