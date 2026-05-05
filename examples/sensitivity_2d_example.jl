using Revise
using OptimalBath
using GLMakie

function bump(x, a, c)
    return exp(-a * (x - c)^2)
end

function bathymetry(x, y)
    return 1-0.01x +0* 0.3 * bump(x, 0.01, 30) * (bump(y, 0.01, 50))# + bump(y, 0.01, 60))
end

function initial_height(x, y)
    return 2 - 0.05x
end

initial_condition(x, y) = State(initial_height(x, y), 0.0, 0.0)

function determine_objective_indices(grid)
    x_centers, y_centers = cell_centers(grid)
    le(a) = x -> x < a
    gt(a) = x -> x > a
    return CartesianIndices((findfirst(gt(50), x_centers):findlast(le(75), x_centers),
                             findfirst(gt(40), y_centers):findlast(le(60), y_centers)))
end

function create_objectives(grid::Grid{2})
    return Objectives(objective_indices=determine_objective_indices(grid),
                      interior_objective=1e-4Mass(),
                      design_indices=CartesianIndices(grid.N .+ 1),
                      regularization=0.01SoftL1(20.0))
end
