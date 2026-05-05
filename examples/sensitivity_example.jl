function bump(x, c)
    return exp(-0.01 * (x - c)^2)
end

function bumps(x, bump_centers)
    return sum(bump_centers) do c
        bump.(x, c)
    end
end

function slope(x)
    return 1 .- 0.01x
end

function initial_bathymetry(x, bump_centers)
    return slope(x) + 0.5 * bumps(x, bump_centers)
end

function initial_condition(x)
    if x < 30.0
        return State(2-0.04x, 0.0)
    else
        return State(1-0.04*30 - 0.2 * (x - 30.0), 0.0)
    end
end

function create_objectives(grid::Grid{1})
    N = grid.N[1]
    return Objectives(objective_indices=3N÷4:N, interior_objective=0.00125Mass(), regularization=0.9SoftL1(10.0) + 0.1SoftTV(10.0))
end