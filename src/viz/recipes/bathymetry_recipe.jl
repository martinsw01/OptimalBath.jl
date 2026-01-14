@userplot BathometryPlot
@recipe function f(anim::BathometryPlot)
    x, B = anim.args

    seriestype --> :path
    fillrange --> -1.#0.#H_ylim[1]
    seriescolor --> :brown
    # c --> :brown
    # alpha --> 0.5
    label --> "Bathymetry B"
    # label --> "Bathymetry " * L"B"
    linewidth --> 0.5

    [x], [B]
end