export plot_gradient

function plot_gradient(g, cell_faces)
    @assert size(g) == size(cell_faces)
    plot(cell_faces, g, label="Gradient", xlabel="x", ylabel="∂J/∂β")
end

function plot_gradient(g, cell_faces, U0, b)
    @assert size(g) == size(cell_faces)

    H = height.(U0.U)
    P = momentum.(U0.U)
    HH = combine_matrices(H', H')[1,:]
    PP = combine_matrices(P', P')[1,:]
    x = sort([cell_faces[1:end-1]; cell_faces[2:end]])
    H_lim = calc_ylim(extrema(H)..., 0.1)
    # @show H_lim
    # H_lim = (min(H_lim[1], minimum(g)), max(H_lim[2], maximum(g)))
    # @show H_lim

    background_plot = plot()
    gradient_plot = twinx(background_plot)

    plot!(background_plot, WaterAnim((x, HH, PP, "Initial condition", H_lim, (0, 1))), colorbar=false, legend=:topright, ylabel="Height")
    plot!(background_plot, BathometryPlot((cell_faces, b, "Bathymetry")))#, seriescolor="#aa5c27")

    hline!(gradient_plot, [0], linestyle=:dash, color=:black, label=nothing)
    plot!(gradient_plot, cell_faces, g, label="Gradient", xlabel="x", ylabel="∂J/∂β", color="#000000", linewidth=2)
    
    background_plot[1][:yaxis][:mirror] = true
    background_plot[2][:yaxis][:mirror] = false
    return background_plot
end