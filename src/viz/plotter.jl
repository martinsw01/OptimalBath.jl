using Plots
function plot_solution(W)
    W1 = _height.(W)
    W2 = _adjoint_momentum.(W)
    # J = compute_gradient(W2, Δt)
    plot(
        heatmap(W1', title="\$\\lambda_1\$", xlabel="Space", ylabel="Time"),
        # heatmap(W1', title="\$\\lambda_1\$", xlabel="Space", ylabel="Time", clims = (-1, 1)),
        # heatmap(log.(abs.(W1') .+ 1) .* sign.(W1'), title="\$\\lambda_1\$", xlabel="Space", ylabel="Time"),
        # heatmap(log.(abs.(W1') .+ 1), title="\$\\lambda_1\$", xlabel="Space", ylabel="Time"),
        heatmap(W2', title="\$\\lambda_2\$", xlabel="Space", ylabel="Time"),
        # heatmap(W2', title="\$\\lambda_2\$", xlabel="Space", ylabel="Time", clims= (-1, 1)),
        # heatmap(log.(abs.(W2') .+ 1) .* sign.(W2'), title="\$\\lambda_2\$", xlabel="Space", ylabel="Time"),
        # heatmap(log.(abs.(W2') .+ 1), title="\$\\lambda_2\$", xlabel="Space", ylabel="Time"),
        # bar(J, title="Gradient", xlabel="Space", ylabel="dJ/dβ"),
        # layout = (1, 3), size=2 .* (1600, 600)
        layout = (2, 1), size=2 .* (1200, 1200)
    )
end