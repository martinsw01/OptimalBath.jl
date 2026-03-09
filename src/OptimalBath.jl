module OptimalBath

include("interfaces/states.jl")
include("interfaces/gradients.jl")
include("interfaces/forward_solvers.jl")

include("PrimalSWE.jl")
include("AdjointSWE.jl")
include("gradients/Gradients.jl")
include("solver.jl")
include("objectives.jl")

end
