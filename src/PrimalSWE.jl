using SinFVM, ElasticArrays

"""
    SinFVMPrimalSWEProblem

Wrapper around SinFVM. Contains problem definition and solver parameters.
"""
struct SinFVMPrimalSWEProblem <: PrimalSWEProblem
    _make_backend
    initial_bathymetry
    _grid
    _reconstruction
    _timestepper
    u0
    T
    function SinFVMPrimalSWEProblem(N,
                              u0,
                              T;
                              domain = [0.0 1.0],
                              _make_backend=SinFVM.make_cpu_backend,
                              initial_bathymetry=zeros(N + 1),
                              _grid=SinFVM.CartesianGrid(N; gc=2, boundary=SinFVM.WallBC(), extent = domain),
                              _reconstruction=SinFVM.LinearReconstruction(1.),
                              _timestepper=SinFVM.RungeKutta2())
        @assert length(initial_bathymetry) == N + 1 "Bathymetry must have length N+1=$(N + 1) ≠ $(length(initial_bathymetry))"
        return new(_make_backend,
                   initial_bathymetry,
                   _grid,
                   _reconstruction,
                   _timestepper,
                   u0,
                   T)
    end       
end

function _create_simulator(problem, β)
    FloatType = eltype(β)
    backend = problem._make_backend(FloatType)
    b = vcat(zeros(FloatType, 2),
             problem.initial_bathymetry .+ β,
             zeros(FloatType, 2))
    bathymetry = SinFVM.BottomTopography1D(b, backend, problem._grid)
    equation = SinFVM.ShallowWaterEquations1D(bathymetry)
    numericalflux = SinFVM.CentralUpwind(equation)
    conserved_system = SinFVM.ConservedSystem(backend, problem._reconstruction, numericalflux, equation, problem._grid, [SinFVM.SourceTermBottom()])
    simulator = SinFVM.Simulator(backend, conserved_system, problem._timestepper, problem._grid, cfl=0.2)
    return simulator
end

function recording_callback(problem::SinFVMPrimalSWEProblem,
                            U0::States{Average, Elevation, T}, t0) where T
    U = States{Average, Elevation}(ElasticMatrix(reshape(U0.U, :, 1)))
    t = ElasticVector([t0])

    record_state = create_callback(problem) do U_n, t_n, Δt
        append!(U.U, U_n.U)
        append!(t, t_n)
    end

    return record_state, U, t
end

function create_callback(f, ::SinFVMPrimalSWEProblem)
    function callback(t_n, simulator)
        U = SinFVM.current_interior_state(simulator)
        Δt = SinFVM.current_timestep(simulator)
        f(States{Average, Elevation}(U), t_n, Δt)
    end
    return callback
end

function solve_primal(problem::SinFVMPrimalSWEProblem, β)
    FloatType = eltype(β)
    t0 = zero(FloatType)
    callback, U, t = recording_callback(problem, problem.u0, t0)
    solve_primal(problem, β, callback)
    x = CellFaces(SinFVM.cell_faces(problem._grid))
    return U, t, x
end

function solve_primal(problem::SinFVMPrimalSWEProblem, β, callback)
    simulator = _create_simulator(problem, β)

    SinFVM.set_current_state!(simulator, problem.u0)

    SinFVM.simulate_to_time(simulator, problem.T; callback=callback)
end

function compute_Δx(problem::SinFVMPrimalSWEProblem)
    return SinFVM.compute_dx(problem._grid)
end

function initial_state(problem::SinFVMPrimalSWEProblem)
    return problem.u0
end

using SinFVM: Reconstruction, for_each_cell, B_cell, B_face_left, B_face_right, minmod_slope, AllPracticalSWE, for_each_inner_cell
import SinFVM: reconstruct!

"""
    BathymetryHandlingReconstruction()
A linear reconstruction similar to SinFVM.LinearReconstruction, but caps heights and momentum to zero instead of adjusting the slope.
"""
struct BathymetryHandlingReconstruction <: Reconstruction end

function clip_to_zero(U, b)
    h = height(U)
    if h > b
        return State(h - b, momentum(U))
    else
        return zero(U)
    end
end

function reconstruct_cell(U, B_left, B_right)
    slope = minmod_slope(U, U, U, 1)
    U_left = U .- 0.5 .* slope
    U_right = U .+ 0.5 .* slope
    return clip_to_zero(U_left, B_left), clip_to_zero(U_right, B_right)
end

function SinFVM.reconstruct!(backend, ::BathymetryHandlingReconstruction, output_left, output_right, input_conserved, grid::SinFVM.Grid, eq::AllPracticalSWE, direction)
    @assert grid.ghostcells[1] > 1

    for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright

        B_left = B_face_left(eq.B, imiddle, direction)
        B_right = B_face_right(eq.B, imiddle, direction)
        U_left, U_right = reconstruct_cell(input_conserved[imiddle], B_left, B_right)

        output_left[imiddle], output_right[imiddle] = U_left, U_right
    end

    return nothing
end

using SinFVM: SourceTerm, ConservedSystem, Direction, compute_dx
import SinFVM: evaluate_directional_source_term!


"""
    DryFrontTerm()

A bottom topography source term corresponding to `BathymetryHandlingReconstruction`. Similar to the one used in SinFVM,
except that it balances the fluxes also at dry fronts, where the water height goes to zero.
"""
struct DryFrontTerm <: SourceTerm end

function SinFVM.evaluate_directional_source_term!(::DryFrontTerm, output, current_state, cs::ConservedSystem, dir::Direction)
    # {right, left}_buffer is (h, hu)
    # output and current_state is (w, hu)
    dx = compute_dx(cs.grid, dir)
    output_momentum = output.hu
    B = cs.equation.B 
    g = cs.equation.g
    h_right = cs.right_buffer.h
    h_left  = cs.left_buffer.h
    for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        B_right = B_face_right( B, imiddle, dir)
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