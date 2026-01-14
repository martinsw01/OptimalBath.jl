@userplot WaterAnim
@recipe function f(anim::WaterAnim)
    x, H, UH, name, H_ylim, UH_ylim = anim.args
    
    ylims --> H_ylim
    # ylim --> H_ylim
    # ylim --> (0., H_ylim[2])
    legend --> :topleft
    # legendcolumns --> length(names)
    label --> "w = h + B"#name
    # label --> L"w = h + B"#name

    linewidth --> 0.5

    yguide --> "Height above reference (m)"
    # yguide --> "Height above reference (\\mathrm{m})"
    # ylabel --> "Height above reference (\\mathrm{m})"
    # ylabel --> "Height above reference " * L"(\mathrm{m})"
    xguide --> "Position (m)"
    # xguide --> "Position (\\mathrm{m})"
    # xlabel --> "Position (\\mathrm{m})"
    # xlabel --> "Position " * L"(\mathrm{m})"

    seriestype --> :path
    fillrange --> 0.#H_ylim[1]
    fill_z --> abs.(UH)

    colorbar_title --> " \nMagnitude of momentum (m^2s^{-1})"
    # colorbar_title --> " \nMagnitude of momentum (\\mathrm{m}^2\\mathrm{s}^{-1})"
    # colorbar_title --> " \nMagnitude of momentum " * L"(\mathrm{m}^2\mathrm{s}^{-1})"
    right_margin --> 10Plots.mm
    top_margin --> 2Plots.mm
    # dpi --> 300
    # Define the color gradient
    seriescolor --> :blues
    # c --> :blues
    clims --> UH_ylim

    [x], [H]
end