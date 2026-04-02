module OptimalBath

include("interfaces/states.jl")
include("interfaces/gradients.jl")
include("interfaces/forward_solvers.jl")
include("grids.jl")
include("solver_options.jl")

include("PrimalSWE.jl")
include("gradients/gradients.jl")
include("solver.jl")
include("objectives.jl")

end
