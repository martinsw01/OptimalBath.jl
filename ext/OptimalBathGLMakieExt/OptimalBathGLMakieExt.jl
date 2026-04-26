module OptimalBathGLMakieExt

using OptimalBath
using GLMakie

include("water_1d_plots.jl")
include("water_1d_anims.jl")
include("gradients_1d.jl")
include("objective_convergence_plots.jl")
include("optimization_animations.jl")
include("gradients_2d.jl")

end