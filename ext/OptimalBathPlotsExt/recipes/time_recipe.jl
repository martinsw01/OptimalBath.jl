@userplot TimeAnim
@recipe function f(tp::TimeAnim)
    t_n, T, x_L, x_R, (_, ymax) = tp.args

    label --> ["" "\$t=$(round(t_n, digits=2))\$"]
    linewidth --> 10
    # color --> [:lightgrey :grey]
    seriescolor --> [:lightgrey :grey]

    [[x_L, x_R], [x_L, (x_R - x_L) * t_n / T + x_L]],
    [[ymax, ymax], [ymax, ymax]]
end