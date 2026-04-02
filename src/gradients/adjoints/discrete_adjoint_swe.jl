export DiscreteAdjointSWE, TestDiscreteAdjoint, StepWiseTestDiscreteAdjoint

using ForwardDiff: jacobian, gradient
using VolumeFluxes: CentralUpwind, ShallowWaterEquations1D, XDIR
using StaticArrays: @SMatrix, SMatrix

import OptimalBath: solve_adjoint
using .OptimalBath: compute_ghost_cell, AdjointSWE
using .OptimalBath: Grid, for_each_left_boundary_directional_stencil, for_each_interior_directional_stencil, for_each_right_boundary_directional_stencil


struct DiscreteAdjointSWE{PrimalSolver<:VolumeFluxesSolver, Dims} <: AdjointSWE
    primal::PrimalSolver
    grid::Grid{Dims}
    function DiscreteAdjointSWE(primal)
        grid = primal.problem.grid
        return new{typeof(primal), length(grid.Δx)}(primal, grid)
    end
end

function primal_eq(da::DiscreteAdjointSWE)
    return da.primal.simulator.system.equation
end

function primal_numerical_flux(da::DiscreteAdjointSWE)
    return da.primal.simulator.system.numericalflux
end

function flux_left_grad_center(Ul, Uc, dir, da::DiscreteAdjointSWE)
    return jacobian(Uc) do U
        eq = primal_eq(da)
        F = primal_numerical_flux(da)(eq, Ul, U, Val(dir))[1]
        return F
    end
end

function flux_right_grad_center(Uc, Ur, dir, da::DiscreteAdjointSWE)
    return jacobian(Uc) do U
        eq = primal_eq(da)
        F = primal_numerical_flux(da)(eq, U, Ur, Val(dir))[1]
        return F
    end
end


function time_step_source(Λ, U_next, U_prev, wave_speed, CFL, Δx, Δt, ∂a∂U)
    τ = - CFL * Δx / wave_speed^2 * ∂a∂U
    # Allocation free version of `dot(Λ, U_next - U_prev)/Δt`
    τ * sum(zip(Λ, U_next, U_prev)) do (Λ_j, U_next_j, U_prev_j)
        ∂U∂t = (U_next_j - U_prev_j)
        return Λ_j' * ∂U∂t
    end / Δt
end

function desingularize(h, da::DiscreteAdjointSWE)
    return OptimalBath.desingularize(h, da.primal)
end

function desingularize(h, p, da::DiscreteAdjointSWE)
    return OptimalBath.desingularize(h, p, da.primal)
end

function depth_cutoff(primal_solver)
    return primal_solver.simulator.system.equation.depth_cutoff
end

function compute_max_abs_eigval(U::State, da::DiscreteAdjointSWE)
    h, p = U
    if h < depth_cutoff(da.primal)
        return zero(h)
    else
        u = desingularize(h, p, da)
        c = sqrt(9.81*h)
        return abs(u) + abs(c)
    end
end

function determine_time_step_index(U, da::DiscreteAdjointSWE)
    findmax(U) do U_j
        return compute_max_abs_eigval(U_j, da)
    end |> last
end

function compute_timestep_correction(Λ_next, U_prev, U_next, Δt)
    return Λ_next' * (U_next - U_prev) / Δt
end

function compute_timestep_gradient(CFL, Δx, Ui, da::DiscreteAdjointSWE)
    gradient(Ui) do U
        a = compute_max_abs_eigval(U, da)
        return CFL * Δx / a
    end
end

function compute_timestep_source(Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, da::DiscreteAdjointSWE)
    i = determine_time_step_index(U_prev, da)
    τ = compute_timestep_gradient(CFL, Δx, U_prev[i], da)
    μ_step = compute_timestep_correction(Λ_next, U_prev, U_next, Δt)
    J_step = compute_objective_step(U_prev, Δx, objectives)
    Δμ = μ_step - μ_final
    ΔJ = J_step - J_final
    return i, (Δμ + ΔJ) * τ
end

function add_timestep_source!(Λ_prev, Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, da::DiscreteAdjointSWE)
    i, source = compute_timestep_source(Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, da)
    Λ_prev[i] += source
end

function add_objective_timestep_source!(Λ_prev, U_prev, J_final, Δx, CFL, objectives, da::DiscreteAdjointSWE)
    i = determine_time_step_index(U_prev, da)
    τ = compute_timestep_gradient(CFL, Δx, U_prev[i], da)
    J_step = compute_objective_step(U_prev, Δx, objectives)
    ΔJ = J_step - J_final
    Λ_prev[i] += ΔJ * τ
end

function add_flux_jvp_left_boundary!(Λ1, Λ2, Uc, Ur, Δt, Δx, dir, da::DiscreteAdjointSWE)
    Ul = compute_ghost_cell(Uc, dir)
    
    A = @SMatrix [1 0; 0 -1]
    dFldU_ghost = flux_right_grad_center(Ul, Uc, dir, da) * A
    dFldU = flux_left_grad_center(Ul, Uc, dir, da)
    dFrdU = flux_right_grad_center(Uc, Ur, dir, da)
    center = - (dFrdU - dFldU_ghost - dFldU) / Δx * Δt
    right = dFrdU / Δx * Δt
    return center' * Λ1 + right' * Λ2
end


function add_flux_jvp_interior!(Λl, Λc, Λr, Ul, Uc, Ur, Δt, Δx, dir, da::DiscreteAdjointSWE)
    dFldU = flux_left_grad_center(Ul, Uc, dir, da)
    dFrdU = flux_right_grad_center(Uc, Ur, dir, da)

    left = -dFldU / Δx * Δt
    center = - (dFrdU - dFldU) / Δx * Δt
    right = dFrdU / Δx * Δt
    return left' * Λl + center' * Λc + right' * Λr
end

function add_flux_jvp_right_boundary!(Λl, Λc, Ul, Uc, Δt, Δx, dir, da::DiscreteAdjointSWE)
    Ur = compute_ghost_cell(Uc, dir)
    
    dFldU = flux_left_grad_center(Ul, Uc, dir, da)
    dFrdU = flux_right_grad_center(Uc, Ur, dir, da)
    A = @SMatrix [1 0; 0 -1]
    dFrdU_ghost = flux_left_grad_center(Uc, Ur, dir, da) * A
    left = -dFldU / Δx * Δt
    center = (dFldU - dFrdU_ghost - dFrdU) / Δx * Δt
    return left' * Λl + center' * Λc
end

function add_flux_jvp!(Λ, Λ_pp, U, n, Δt, dir, grid, da::DiscreteAdjointSWE)
    Δx = grid.Δx[dir]
    for_each_left_boundary_directional_stencil(dir, grid) do center, right
        Λ[center, n-1] += add_flux_jvp_left_boundary!(Λ_pp[center], Λ_pp[right],
                                                      U[center, n-1], U[right, n-1],
                                                      Δt, Δx, dir, da)
    end

    for_each_interior_directional_stencil(dir, grid) do left, center, right
        Λ[center, n-1] += add_flux_jvp_interior!(Λ_pp[left], Λ_pp[center], Λ_pp[right],
                                                 U[left, n-1], U[center, n-1], U[right, n-1],
                                                 Δt, Δx, dir, da)
    end

    for_each_right_boundary_directional_stencil(dir, grid) do left, center
        Λ[center, n-1] += add_flux_jvp_right_boundary!(Λ_pp[left], Λ_pp[center],
                                                       U[left, n-1], U[center, n-1],
                                                       Δt, Δx, dir, da)
    end
end

function add_objective_source!(Λ, U, Δt, Δx, objectives, ::DiscreteAdjointSWE)
    OptimalBath.add_objective_source!(Λ, U, Δt, Δx, objectives)
end

function compute_objective_step(U, Δx, objectives::Objectives)
    indices = objectives.objective_indices
    objective = objectives.interior_objective
    Jn = sum(objective_density.(objective, @view U[indices])) * Δx
    return Jn
end

function bottom_source_type(::DiscreteAdjointSWE{<:PrimalSWESolver{R, TS, BottomSourceType}}) where {R, TS, BottomSourceType}
    return BottomSourceType
end

function add_bottom_source!(Λ, Λ_pp, n, t, Δx, b, da::DiscreteAdjointSWE)
    add_bottom_source!(Λ, Λ_pp, n, t, Δx, b, bottom_source_type(da))
end

function add_bottom_source!(Λ, Λ_pp, n, t, Δx, b, ::Type{DefaultBathymetrySource})
    Δt = t[n] - t[n-1]
    for j in axes(Λ, 1)
        Δb = b[j+1] - b[j]
        S12 = -9.81 * Δb * Δt / Δx
        Λ[j, n-1] += State(S12 * momentum(Λ_pp[j]), 0)
    end
end

function zero_momentum_state(Λ::State{2})
    return State(height(Λ), 0)
end

function adjoint_pre_proc_step!(Λ_pp, Λ, U, n, grid, da)
    OptimalBath.for_each_cell(grid) do j
        if height(U[j, n]) < depth_cutoff(da.primal)
            Λ_pp[j] = zero_momentum_state(Λ[j, n])
        else
            Λ_pp[j] = Λ[j, n]
        end
        Λ[j, n-1] = Λ_pp[j]
    end
end


function time_steps(U, ::Grid{Dims}) where Dims
    return size(U, Dims+1)
end

@views function solve_adjoint(Λ_end, U::AverageDepthStates, objectives::Objectives, b, t, Δx, da::DiscreteAdjointSWE)
    U = U.U
    Λ = similar(U)
    Λ_pp = similar(Λ_end)

    grid = da.grid

    Δt = t[end] - t[end-1]
    μ_final = compute_timestep_correction(Λ_end, U[:, end-1], U[:, end], Δt)
    J_final = compute_objective_step(U[:, end-1], Δx, objectives)

    M = time_steps(U, grid)
    Λ[:, end] .= Λ_end
    dir = 1
    for n in M:-1:2
        Δt = t[n] - t[n-1]
        adjoint_pre_proc_step!(Λ_pp, Λ, U, n, grid, da)
        add_flux_jvp!(Λ, Λ_pp, U, n, Δt, dir, grid, da)
        add_bottom_source!(Λ, Λ_pp, n, t, Δx, b, da)
        add_objective_source!(Λ[:, n-1], U[:, n-1], Δt, Δx, objectives, da)
        if n < M
            add_timestep_source!(Λ[:, n-1], Λ[:, n], U[:, n], U[:, n-1], μ_final, J_final, Δt, Δx, 0.25, objectives, da)
        end
    end
    return Λ
end

