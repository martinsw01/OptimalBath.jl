function run_SinFVM_simulation(T, N, u0, b = zeros(N + 1))
    backend = make_cpu_backend(eltype(b))

    grid = SinFVM.CartesianGrid(N; gc=2, boundary=SinFVM.WallBC())
    
    bathymetry = SinFVM.BottomTopography1D(vcat(zeros(eltype(b), 2), b, zeros(eltype(b), 2)), 
                                           backend, grid)
    equation = SinFVM.ShallowWaterEquations1D(bathymetry)
    reconstruction = SinFVM.LinearReconstruction(1.)
    numericalflux = SinFVM.CentralUpwind(equation)
    timestepper = SinFVM.RungeKutta2()
    conserved_system = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [SinFVM.SourceTermBottom()])
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=1.) # Should be cfl=0.2, but it is not yet implemented in homemade_conslaws
    
    x = SinFVM.cell_centers(grid)
    initial = [eltype(b).(U) for U in u0.(x)]
    SinFVM.set_current_state!(simulator, initial)

    U = ElasticMatrix(reshape(initial, :, 1))
    t = ElasticVector([eltype(b)(0.)])

    function collect_state(t_j, simulator)
        append!(U, SinFVM.current_interior_state(simulator))
        append!(t, t_j)
    end

    
    SinFVM.simulate_to_time(simulator, T; callback=collect_state)
    
    return U, t, x
end

# function compute_loss(b::Vector)
#     U, t, x = run_SinFVM_simulation(0.2, 10, u0, b)
#     Δx = x[2] - x[1]

#     H = [h for (h, hu) in U]
    
#     integral = zero(eltype(H))
#     for n in eachindex(t)[1:end-1]
#         Δt = t[n+1] - t[n]
#         integral += 0.5 * (sum(H[:,n]) + sum(H[:,n+1])) * Δt
#     end
#     return integral * Δx

#     reduce(zip(eachcol(H)[1:end-1], eachcol(H)[2:end], t[1:end-1], t[2:end])) do acc, (h1, h2, t1, t2)
#         Δt = t2 - t1
#         acc + 0.5 * sum(h1 + h2) * Δt * Δx
#     end
# end
    

# ForwardDiff.gradient(compute_loss, b)

# U, t, x = run_SinFVM_simulation(0.2, 10, u0)#(x) -> @SVector [1.0 + 0.5*x, 0.0])
# # Δx = x[2] - x[1]
# # b = zeros(length(x) + 1)

# interior_loss_density(U, b) = [one(eltype(U)), zero(eltype(U))]
# terminal_loss_density(U) = zero(U) #@SVector [zero(eltype(U)), one(eltype(U))]

# Λ = solve_adjoint(U, t, Δx, b, interior_loss_density, terminal_loss_density)

# # Λ