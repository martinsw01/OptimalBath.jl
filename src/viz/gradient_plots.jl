export plot_gradient

function plot_gradient(g, cell_faces)
    @assert size(g) == size(cell_faces)
    plot(cell_faces, g, label="Gradient", xlabel="x", ylabel="∂J/∂β")
end