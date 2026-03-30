export VolumeFluxesBackend, VolumeFluxesSolver, MinModSlope

using VolumeFluxes, ElasticArrays

struct VolumeFluxesBackend <: SolverBackend end

function build_solver(spec::SolverSpec{PP, VolumeFluxesBackend, SO}, ::Type{FloatType}) where {PP, SO, FloatType}
    return VolumeFluxesSolver(spec, FloatType)
end

struct MinModSlope <: LinearReconstruction end

function to_VF(::MinModSlope)
    return VolumeFluxes.LinearReconstruction()
end

function to_VF(::NoReconstruction)
    return VolumeFluxes.NoReconstruction()
end

function to_VF(::ForwardEuler)
    return VolumeFluxes.ForwardEulerStepper()
end

function to_VF(::RK2)
    return VolumeFluxes.RungeKutta2()
end

function to_VF(::DefaultBathymetrySource)
    return VolumeFluxes.SourceTermBottom()
end

const InitialStates{FloatType} = States{Average, Elevation, FloatType, 1, Vector{State{FloatType}}}

"""
    VolumeFluxesSolver(spec::SolverSpec, float_type)
    VolumeFluxesSolver(problem::PrimalSWEProblem, options::SolverOptions, float_type)
    VolumeFluxesSolver(problem::PrimalSWEProblem, reconstruction, timestepper, bathymetry_source, float_type)
A wrapper around VolumeFluxes's SWE solver.
"""
struct VolumeFluxesSolver{Simulator, Problem, R, TS, BS, FloatType} <: PrimalSWESolver{R, TS, BS}
    simulator::Simulator
    problem::Problem
    function VolumeFluxesSolver(spec::SolverSpec{PP, VolumeFluxesBackend, SO}, ::Type{FloatType}) where {PP, SO, FloatType}
        problem = spec.problem
        options = spec.solver_options
        return VolumeFluxesSolver(problem, options, FloatType)
    end
    function VolumeFluxesSolver(problem::PrimalSWEProblem, options::SolverOptions{R, TS, BS}, ::Type{FloatType}) where {R, TS, BS, FloatType}
        return VolumeFluxesSolver(problem, options.reconstruction, options.timestepper, options.bathymetry_source, FloatType)
    end
    function VolumeFluxesSolver(problem::PrimalSWEProblem, reconstruction::R, timestepper::TS, bathymetry_source::BS, ::Type{FloatType}) where {R, TS, BS, FloatType}
        simulator = create_simulator(problem, reconstruction, timestepper, bathymetry_source, FloatType)
        return new{typeof(simulator), typeof(problem), R, TS, BS, FloatType}(simulator, problem)
    end
end

function desingularize(h, solver::VolumeFluxesSolver)
    return VolumeFluxes.desingularize(solver.simulator.system.equation, h)
end

function desingularize(h, p, solver::VolumeFluxesSolver)
    return VolumeFluxes.desingularize(solver.simulator.system.equation, h, p)
end

function create_grid(problem, reconstruction)
    return _create_grid(problem.N, problem.domain, reconstruction)
end

function _create_grid(N, domain, ::LinearReconstruction)
    return _create_grid(N, domain, 2)
end

function _create_grid(N, domain, ::NoReconstruction)
    return _create_grid(N, domain, 1)
end

function _create_grid(N, domain, gc)
    return VolumeFluxes.CartesianGrid(N; gc=gc, boundary=VolumeFluxes.WallBC(), extent = domain)
end

@views function extrapolate_bathymetry(initial_bathymetry, float_type, gc)
    b = similar(initial_bathymetry, float_type, length(initial_bathymetry) + 2*gc)
    b[gc+1:end-gc] .= initial_bathymetry
    update_bc!(b, gc)
    # b[1:gc] .= initial_bathymetry[2*gc:-1:gc+1]
    # b[end-gc+1:end] .= initial_bathymetry[end-2*gc+1:end-gc]
    return b
end

@views function update_bc!(bathymetry, gc)
    bathymetry[1:gc] .= bathymetry[2gc+1:-1:gc+2]
    bathymetry[end-gc+1:end] .= bathymetry[end-gc-1:-1:end-2gc]
end


function extrapolate_bathymetry(initial_bathymetry, float_type, ::NoReconstruction)
    return extrapolate_bathymetry(initial_bathymetry, float_type, 1)
end

function extrapolate_bathymetry(initial_bathymetry, float_type, ::LinearReconstruction)
    return extrapolate_bathymetry(initial_bathymetry, float_type, 2)
end

function make_bathymetry(problem, reconstruction, float_type)
    return extrapolate_bathymetry(problem.initial_bathymetry, float_type, reconstruction)
end


function construct_equation(problem, reconstruction, backend, grid)
    b = make_bathymetry(problem, reconstruction, backend.realtype)
    bathymetry = VolumeFluxes.BottomTopography1D(b, backend, grid)
    equation = VolumeFluxes.ShallowWaterEquations1D(bathymetry)
    return equation
end

function create_simulator(problem, reconstruction, timestepper, bathymetry_source, float_type)
    cpu_backend = VolumeFluxes.make_cpu_backend(float_type)
    grid = create_grid(problem, reconstruction)
    equation = construct_equation(problem, reconstruction, cpu_backend, grid)
    numericalflux = VolumeFluxes.CentralUpwind(equation)
    conserved_system = VolumeFluxes.ConservedSystem(cpu_backend, to_VF(reconstruction), numericalflux, equation, grid, [to_VF(bathymetry_source)])
    simulator = VolumeFluxes.Simulator(cpu_backend, conserved_system, to_VF(timestepper), grid)
    return simulator
end

function recording_averages_callback(solver, t0)
    U0 = VolumeFluxes.current_interior_state(solver.simulator)
    U = States{Average, Elevation}(ElasticMatrix(reshape(U0, :, 1)))
    t = ElasticVector([t0])

    record_state = create_callback(solver) do U_n, t_n, Δt
        append!(U.U, U_n.U)
        append!(t, t_n)
    end

    return record_state, U, t
end


function recording_reconstructions_callback(simulator, t0)
    Ul0 = simulator.system.left_buffer[3:end-2]
    Ur0 = simulator.system.right_buffer[3:end-2]
    Ul = States{Left, Depth}(ElasticMatrix(reshape(Ul0, :, 1)))
    Ur = States{Right, Depth}(ElasticMatrix(reshape(Ur0, :, 1)))
    t = ElasticVector([t0])
    function record_state(t_n, simulator)
        append!(Ul.U, simulator.system.left_buffer[3:end-2])
        append!(Ur.U, simulator.system.right_buffer[3:end-2])
        append!(t, t_n)
    end
    return record_state, (Ul, Ur), t
end

function recording_callback(solver::VolumeFluxesSolver{S,P, NoReconstruction, TS, BS, F}, t0) where {S, P, TS, BS, F}
    return recording_averages_callback(solver, t0)
end

function recording_callback(solver::VolumeFluxesSolver{S,P, R, TS, BS, F}, t0) where {S, P, R<:LinearReconstruction, TS, BS, F}
    return recording_averages_callback(solver, t0) #recording_reconstructions_callback(solver.simulator, t0)
end

function create_callback(f, ::VolumeFluxesSolver)
    function callback(t_n, simulator)
        U = VolumeFluxes.current_interior_state(simulator)
        Δt = VolumeFluxes.current_timestep(simulator)
        f(States{Average, Elevation}(U), t_n, Δt)
    end
    return callback
end

function ghost_cells(simulator)
    return simulator.system.grid.ghostcells[1]
end

function get_bathymetry(simulator)
    simulator.system.equation.B.B
end

function get_bathymetry(solver::VolumeFluxesSolver)
    gc = ghost_cells(solver.simulator)
    return @views get_bathymetry(solver.simulator)[gc+1:end-gc]
end

@views function update_bathymetry!(simulator, problem, β)
    gc = ghost_cells(simulator)
    b = get_bathymetry(simulator)
    b[gc+1:end-gc] .= problem.initial_bathymetry .+ β
    update_bc!(b, gc)
end

function set_initial_state!(simulator, problem, β)
    U0 = problem.U0.U
    U0_volume_fluxes = VolumeFluxes.current_interior_state(simulator)
    for i in eachindex(U0)
        β_i = 0.5 * (β[i] + β[i+1])
        U0_volume_fluxes[i] = U0[i] + State(β_i, 0)
    end
    VolumeFluxes.update_bc!(simulator, VolumeFluxes.current_state(simulator))
end

function reset_time!(simulator)
    simulator.t[1] = zero(eltype(simulator.t))
end

function reset_simulator!(simulator, problem, β)
    update_bathymetry!(simulator, problem, β)
    set_initial_state!(simulator, problem, β)
    reset_time!(simulator)
end

function solve_primal(solver::VolumeFluxesSolver, β)
    reset_simulator!(solver.simulator, solver.problem, β)
    
    t0 = zero(eltype(β))
    callback, U, t = recording_callback(solver, t0)

    VolumeFluxes.simulate_to_time(solver.simulator, solver.problem.T; callback=callback)

    x = VolumeFluxes.cell_faces(solver.simulator.system.grid)
    return U, t, x
end

function solve_primal(solver::VolumeFluxesSolver, β, callback)
    reset_simulator!(solver.simulator, solver.problem, β)
    VolumeFluxes.simulate_to_time(solver.simulator, solver.problem.T; callback=callback)
end

function compute_Δx(solver::VolumeFluxesSolver)
    return VolumeFluxes.compute_dx(solver.simulator.system.grid)
end

function initial_state(solver::VolumeFluxesSolver)
    return solver.problem.U0
end

using VolumeFluxes: for_each_cell, B_cell, B_face_left, B_face_right, minmod_slope, AllPracticalSWE, for_each_inner_cell
import VolumeFluxes: reconstruct!

"""
    BathymetryHandlingReconstruction()
A linear reconstruction similar to VolumeFluxes.LinearReconstruction, but caps heights and momentum to zero instead of adjusting the slope.
"""
struct BathymetryHandlingReconstruction <: VolumeFluxes.Reconstruction end

function convert_to_depth(U, b, ε)
    h, p = U
    if is_interface_dry(U, b, ε)
        return zero(U)
    else
        return State(h - b, p)
    end
end

function is_dry(U, b, ε)
    return abs(height(U) - b) < ε
end

function is_interface_dry(U, b, ε)
    return height(U) < b + ε
end

function reconstruct_cell(Ul, Uc, Ur, B_left, B_right, ε, compute_slope)
    if is_dry(Uc, 0.5 * (B_left + B_right), ε)
        return zero(Ul), zero(Ur)
    end
    if is_interface_dry(Uc, B_left, ε)
        return zero(Ul), convert_to_depth(Uc, B_right, ε)
    end
    if is_interface_dry(Uc, B_right, ε)
        return convert_to_depth(Uc, B_left, ε), zero(Ur)
    end

    slope = compute_slope(Ul, Uc, Ur)

    U_left = Uc .- 0.5 .* slope
    U_right = Uc .+ 0.5 .* slope

    return convert_to_depth(U_left, B_left, ε), convert_to_depth(U_right, B_right, ε)
end

function VolumeFluxes.reconstruct!(backend, ::BathymetryHandlingReconstruction, output_left, output_right, input_conserved, grid::VolumeFluxes.Grid, eq::AllPracticalSWE, direction)
    @assert grid.ghostcells[1] > 1

    compute_slope = (Ul, Uc, Ur) -> minmod_slope(Ul, Uc, Ur, 1.)

    for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        B_left = B_face_left(eq.B, imiddle, direction)
        B_right = B_face_right(eq.B, imiddle, direction)
        U_left, U_right = reconstruct_cell(input_conserved[ileft:iright]..., B_left, B_right, eq.depth_cutoff, compute_slope)
        output_left[imiddle], output_right[imiddle] = U_left, U_right
    end

    return nothing
end

using VolumeFluxes: SourceTerm, ConservedSystem, Direction, compute_dx
import VolumeFluxes: evaluate_directional_source_term!


"""
    DryFrontTerm()

A bottom topography source term corresponding to `BathymetryHandlingReconstruction`. Similar to the one used in VolumeFluxes,
except that it balances the fluxes also at dry fronts, where the water height goes to zero.
"""
struct DryFrontTerm <: SourceTerm end

function VolumeFluxes.evaluate_directional_source_term!(::DryFrontTerm, output, current_state, cs::ConservedSystem, dir::Direction)
    # {right, left}_buffer is (h, hu)
    # output and current_state is (w, hu)
    dx = compute_dx(cs.grid, dir)
    output_momentum = output.hu
    B = cs.equation.B 
    g = cs.equation.g
    h_right = cs.right_buffer.h
    h_left  = cs.left_buffer.h
    for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        B_right = B_face_right(B, imiddle, dir)
        B_left  = B_face_left(B, imiddle, dir)

        output_momentum[imiddle] += compute_source_term(h_left[imiddle], h_right[imiddle], B_left, B_right, g, dx, cs.equation.depth_cutoff)
    end
end

function compute_source_term(hl, hr, Bl, Br, g, dx, depth_cutoff)
    if only_right_dry(hl, hr, depth_cutoff)
        return -0.5 * g * hl^2 / dx
    elseif only_left_dry(hl, hr, depth_cutoff)
        return 0.5 * g * hr^2 / dx
    else
        return - g*((Br - Bl)/dx)*((hr + hl)/2.0)
    end
end

function only_left_dry(h_left, h_right, depth_cutoff)
    return h_left < depth_cutoff && h_right > depth_cutoff
end

function only_right_dry(h_left, h_right, depth_cutoff)
    return h_left > depth_cutoff && h_right < depth_cutoff
end


# using VolumeFluxes
# using LogExpFunctions: logsumexp

# struct SmoothLinearReconstruction <: Reconstruction end

# struct VFSmoothLinearReconstruction <: VolumeFluxes.Reconstruction 
#     θ::Float64
#     ε::Float64
#     VFSmoothLinearReconstruction(θ=1.2, ε=1e-1) = new(θ, ε)
# end

# function soft_sign(x, ε=1e-1)
#     return tanh(x / ε)
# end

# function soft_min(a, b, c, ε=1e-1)
#     return -logsumexp((-a/ε, -b/ε, -c/ε)) * ε
# end

# function soft_minmod(a, b, c, ε=1e-1)
#     s = soft_sign(a*b, ε) * soft_sign(a*c, ε)
#     return 0.5 * soft_sign(a, ε) * (1 + s) * soft_min(abs(a), abs(b), abs(c), ε)
# end

# function soft_minmod_slope(left, center, right, θ, ε=1e-1)
#     forward_diff = right .- center
#     backward_diff = center .- left
#     central_diff = (forward_diff .+ backward_diff) ./ 2.0
#     return soft_minmod.(θ .* forward_diff, central_diff, θ .* backward_diff, ε)
# end

# function smooth_minmod end