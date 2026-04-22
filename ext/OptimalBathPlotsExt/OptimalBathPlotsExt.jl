module OptimalBathPlotsExt

using OptimalBath

OptimalBath.plot_gradient(::PlotsBackend) = println("OptimalBathPlotsExt.jl!")

end