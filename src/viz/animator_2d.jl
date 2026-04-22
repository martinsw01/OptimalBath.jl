function plot_bathymetry(bathymetry, grid::Grid{2}, camera_angle)
    plot(cell_faces(grid)..., bathymetry',
         zlims=(0, 2),
         color=cgrad([:brown]),
         camera=camera_angle,
         st=:surface,
         title="Bathymetry",
         xlabel="x",
         ylabel="y",
         zlabel="h",
         legend=false)
end

function plot_water_surface(U, grid::Grid{2})
    h = height.(U)
    surface!(cell_centers(grid)..., h',
             color=:blues,
             clim=(-1, 1))
end

function animate_solution(U, t, bathymetry, grid::Grid{2}; camera_angle=(30, 30))
    anim = @animate for n in eachindex(t)
        plot_bathymetry(bathymetry, grid, camera_angle)
        plot_water_surface(U[:, :, n], grid)
        # title!(plt, "Time: $(round(t[n], digits=2)) s")
    end
    gif(anim, fps=10)
end