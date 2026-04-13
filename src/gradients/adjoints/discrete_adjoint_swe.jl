export DiscreteAdjointSWE, TestDiscreteAdjoint, StepWiseTestDiscreteAdjoint

using ForwardDiff: jacobian, gradient
using VolumeFluxes: CentralUpwind, ShallowWaterEquations1D, XDIR
using StaticArrays: @SMatrix, SMatrix, setindex

import OptimalBath: solve_adjoint
using .OptimalBath: compute_ghost_cell, AdjointSWE, time_frame
using .OptimalBath: Grid, for_each_left_boundary_directional_stencil, for_each_interior_directional_stencil, for_each_right_boundary_directional_stencil
using .OptimalBath: directions


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
        F = primal_numerical_flux(da)(eq, Ul, U, dir)[1]
        return F
    end
end

function flux_right_grad_center(Uc, Ur, dir, da::DiscreteAdjointSWE)
    return jacobian(Uc) do U
        eq = primal_eq(da)
        F = primal_numerical_flux(da)(eq, U, Ur, dir)[1]
        return F
    end
end


function time_step_source(Λ, U_next, U_prev, wave_speed, CFL, Δx, Δt, ∂a∂U)
    τ = - CFL * Δx / wave_speed^2 * ∂a∂U
    τ * Λ_dot_∂U∂t(Λ, U_next, U_prev, Δt)
end

function Λ_dot_∂U∂t(Λ, U_next, U_prev, Δt)
    # Allocation free version of `dot(Λ, U_next - U_prev)/Δt`
    sum(zip(Λ, U_next, U_prev)) do (Λ_j, U_next_j, U_prev_j)
        ∂U∂t = U_next_j - U_prev_j
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

function compute_max_abs_eigval(U::State, dir, da::DiscreteAdjointSWE)
    h = height(U)
    if h < depth_cutoff(da.primal)
        return zero(h)
    else
        p = momentum(U, dir)
        u = desingularize(h, p, da)
        c = sqrt(9.81*h)
        return abs(u) + abs(c)
    end
end

function determine_time_step_index(U, dir, da::DiscreteAdjointSWE)
    findmax(U) do U_j
        return compute_max_abs_eigval(U_j, dir, da)
    end
end

function compute_timestep_correction(Λ_next, U_prev, U_next, Δt)
    Λ_dot_∂U∂t(Λ_next, U_next, U_prev, Δt)
end

function argmin_by_first_component(f, domain)
    min_val = f(first(domain))
    for i in firstindex(domain)+1:lastindex(domain)
        val = f(domain[i])
        if first(val) < first(min_val)
            min_val = val
        end
    end
    return min_val
end

function compute_timestep_gradient(CFL, Δx, Ui, dir, da::DiscreteAdjointSWE)
    gradient(Ui) do U
        a = compute_max_abs_eigval(U, dir, da)
        return CFL * Δx / a
    end
end

function determine_direction_and_timestep_index(U_prev, CFL, grid, da::DiscreteAdjointSWE)
    _, i_min, dir_min = argmin_by_first_component(directions(grid)) do dir
        wave_speed, i = determine_time_step_index(U_prev, dir, da)
        directional_dt = CFL * get_Δx(grid, dir) / wave_speed
        return directional_dt, i, dir
    end
    i_min, dir_min
end

function compute_timestep_source(Λ_next, U_next, U_prev, μ_final, J_final, Δt, grid, CFL, objectives, da::DiscreteAdjointSWE)
    i_min, dir_min = determine_direction_and_timestep_index(U_prev, CFL, grid, da)
    τ = compute_timestep_gradient(CFL, get_Δx(grid, dir_min), U_prev[i_min], dir_min, da)
    μ_step = compute_timestep_correction(Λ_next, U_prev, U_next, Δt)
    J_step = compute_objective_step(U_prev, get_Δx(grid, dir_min), objectives)
    Δμ = μ_step - μ_final
    ΔJ = J_step - J_final
    return i_min, (Δμ + ΔJ) * τ
end

function add_timestep_source!(Λ_prev, Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, da::DiscreteAdjointSWE)
    i, source = compute_timestep_source(Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, da)
    Λ_prev[i] += source
end

function add_objective_timestep_source!(Λ_prev, U_prev, J_final, CFL, objectives, da::DiscreteAdjointSWE)
    i_dir, dir = determine_direction_and_timestep_index(U_prev, CFL, da.grid, da)
    Δx = get_Δx(da.grid, dir)
    τ = compute_timestep_gradient(CFL, Δx, U_prev[i_dir], dir, da)
    J_step = compute_objective_step(U_prev, Δx, objectives)
    ΔJ = J_step - J_final
    Λ_prev[i_dir] += ΔJ * τ
end

using LinearAlgebra: I
function ghost_cell_jvp(U::State{N}, ::Val{dir}) where {N, dir}
    A = SMatrix{N, N, eltype(U)}(I)
    A = setindex(A, -1, 1+dir, 1+dir)
    return A
end

function add_flux_jvp_left_boundary!(Λ1, Λ2, Uc, Ur, Δt, Δx, dir, da::DiscreteAdjointSWE)
    Ul = compute_ghost_cell(Uc, dir)
    
    dFldU_ghost = flux_right_grad_center(Ul, Uc, dir, da) * ghost_cell_jvp(Ul, dir)
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
    dFrdU_ghost = flux_left_grad_center(Uc, Ur, dir, da) * ghost_cell_jvp(Ur, dir)
    left = -dFldU / Δx * Δt
    center = (dFldU - dFrdU_ghost - dFrdU) / Δx * Δt
    return left' * Λl + center' * Λc
end

function add_flux_jvp!(Λ, Λ_pp, U, n, Δt, dir, grid, da::DiscreteAdjointSWE)
    Δx = get_Δx(grid, dir)
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
    Jn = sum(objective_density.(objective, @view U[indices])) * prod(Δx)
    return Jn
end

function bottom_source_type(::DiscreteAdjointSWE{<:PrimalSWESolver{R, TS, BottomSourceType}}) where {R, TS, BottomSourceType}
    return BottomSourceType
end

function add_bottom_source!(Λ, Λ_pp, n, t, b, dir, grid, da::DiscreteAdjointSWE)
    add_bottom_source!(Λ, Λ_pp, n, t, b, dir, grid, bottom_source_type(da))
end

function add_bottom_source!(Λ, Λ_pp, n, t, b, dir, grid, ::Type{DefaultBathymetrySource})
    Δt = t[n] - t[n-1]
    Δx = get_Δx(grid, dir)
    for_each_cell(grid) do j
        Δb = b_at(Right, b, j, dir) - b_at(Left, b, j, dir)
        S12 = -9.81 * Δb * Δt / Δx
        Λ[j, n-1] += setindex(zero(Λ[j, n-1]), S12 * momentum(Λ_pp[j]), 1)
    end
end

function zero_momentum_state(Λ::State)
    return setindex(zero(Λ), height(Λ), 1)
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
    Δx = grid.Δx

    Δt = t[end] - t[end-1]
    M = length(t)
    μ_final = compute_timestep_correction(Λ_end, time_frame(U, M-1), time_frame(U, M), Δt)
    J_final = compute_objective_step(time_frame(U, M-1), Δx, objectives)

    time_frame(Λ, M) .= Λ_end
    dir = 1
    for n in M:-1:2
        Δt = t[n] - t[n-1]
        adjoint_pre_proc_step!(Λ_pp, Λ, U, n, grid, da)
        for dir in directions(grid)
            add_flux_jvp!(Λ, Λ_pp, U, n, Δt, dir, grid, da)
            add_bottom_source!(Λ, Λ_pp, n, t, b, dir, grid, da)
        end
        add_objective_source!(time_frame(Λ, n-1), time_frame(U, n-1), Δt, Δx, objectives, da)
        if n < M
            add_timestep_source!(time_frame(Λ, n-1), time_frame(Λ, n), time_frame(U, n), time_frame(U, n-1), μ_final, J_final, Δt, grid, 0.25, objectives, da)
        end
    end
    return Λ
end

