export DiscreteAdjoint, TestDiscreteAdjoint, StepWiseTestDiscreteAdjoint

using ForwardDiff: jacobian
using SinFVM: CentralUpwind, ShallowWaterEquations1D, XDIR
using StaticArrays: @SMatrix, SMatrix

import OptimalBath: solve_adjoint
using OptimalBath


const _numerical_flux = CentralUpwind(ShallowWaterEquations1D())

abstract type Adjoint end

struct DiscreteAdjoint <: Adjoint
    primal::SinFVMPrimalSWEProblem
end

function compute_ghost_cell(U)
    typeof(U)(U[1], -U[2])
end

function flux_left_grad_center(Ul, Uc)
    return jacobian(Uc) do U
        eq = _numerical_flux.eq
        F = _numerical_flux(eq, Ul, U, XDIR)[1]
        return F
    end
end

function flux_right_grad_center(Uc, Ur)
    return jacobian(Uc) do U
        eq = _numerical_flux.eq
        F = _numerical_flux(eq, U, Ur, XDIR)[1]
        return F
    end
end

function identity_matrix(N)
    I = zeros(N, N)
    for i in 1:N
        I[i, i] = 1.0
    end
    return I
end

I2 = identity_matrix(2)

function ddisplay(A)
    A = round.(A, digits=6)
    println("{{$(A[1, 1]), $(A[1, 2])}, {$(A[2, 1]), $(A[2, 2])}}")
end


function time_step_source(Λ, U_next, U_prev, wave_speed, CFL, Δx, Δt, ∂a∂U)
    τ = - CFL * Δx / wave_speed^2 * ∂a∂U
    # Allocation free version of `dot(Λ, U_next - U_prev)/Δt`
    τ * sum(zip(Λ, U_next, U_prev)) do (Λ_j, U_next_j, U_prev_j)
        ∂U∂t = (U_next_j - U_prev_j)
        return Λ_j' * ∂U∂t
    end / Δt
end

function compute_max_abs_eigval(U::State)
    h, p = U
    u = p / h
    c = sqrt(9.81*h)
    return abs(u) + abs(c)
end

function determine_time_step_index(U)
    findmax(compute_max_abs_eigval, U)
end

# Gradient of eigenvalue wrt U
function eigval_gradient(U::State)
    h, p = U
    p_sgn = sign(p)
    ∂a∂h = -p_sgn * p/h^2 + 0.5 * sqrt(9.81/h)
    ∂a∂p = p_sgn / h
    return State(∂a∂h, ∂a∂p)
end

function compute_timestep_correction(Λ_next, U_prev, U_next, Δt)
    return Λ_next' * (U_next - U_prev) / Δt
end

function compute_timestep_gradient(CFL, Δx, Ui, wave_speed)
    ∂a∂U = eigval_gradient(Ui)
    τ = - CFL * Δx / wave_speed^2 * ∂a∂U
    return τ
end

function compute_timestep_source(Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, local_correction=1)
    wave_speed, i = determine_time_step_index(U_prev)
    τ = compute_timestep_gradient(CFL, Δx, U_prev[i], wave_speed)
    μ_step = compute_timestep_correction(Λ_next, U_prev, U_next, Δt)
    J_step = compute_objective_step(U_prev, Δx, objectives)
    Δμ = local_correction * (μ_step - μ_final)
    ΔJ = J_step - J_final
    return i, (Δμ + ΔJ) * τ
end

function add_timestep_source!(Λ_prev, Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, local_correction=1)
    i, source = compute_timestep_source(Λ_next, U_next, U_prev, μ_final, J_final, Δt, Δx, CFL, objectives, local_correction)
    Λ_prev[i] += source
end

# Only used for the adjoint dot product test
function add_objective_timestep_source!(Λ_prev, U_prev, J_final, Δx, CFL, objectives)
    wave_speed, i = determine_time_step_index(U_prev)
    τ = compute_timestep_gradient(CFL, Δx, U_prev[i], wave_speed)
    J_step = compute_objective_step(U_prev, Δx, objectives)
    ΔJ = J_step - J_final
    Λ_prev[i] += ΔJ * τ
end

function set_flux_jvp_left_boundary!(Λ1, Λ2, Uc, Ur, Δt, Δx, da::DiscreteAdjoint)
    Ul = compute_ghost_cell(Uc)
    
    A = @SMatrix [1 0; 0 -1]
    dFldU_ghost = flux_right_grad_center(Ul, Uc) * A
    dFldU = flux_left_grad_center(Ul, Uc)
    dFrdU = flux_right_grad_center(Uc, Ur)
    center = I2 .- (dFrdU - dFldU_ghost - dFldU) / Δx * Δt
    right = dFrdU / Δx * Δt
    return center' * Λ1 + right' * Λ2
end


function set_flux_jvp_interior!(Λl, Λc, Λr, Ul, Uc, Ur, Δt, Δx, da::DiscreteAdjoint)
    dFldU = flux_left_grad_center(Ul, Uc)
    dFrdU = flux_right_grad_center(Uc, Ur)

    left = -dFldU / Δx * Δt
    center = I2 .- (dFrdU - dFldU) / Δx * Δt
    right = dFrdU / Δx * Δt
    return left' * Λl + center' * Λc + right' * Λr
end

function set_flux_jvp_right_boundary!(Λl, Λc, Ul, Uc, Δt, Δx, da::DiscreteAdjoint)
    Ur = compute_ghost_cell(Uc)
    
    dFldU = flux_left_grad_center(Ul, Uc)
    dFrdU = flux_right_grad_center(Uc, Ur)
    A = @SMatrix [1 0; 0 -1]
    dFrdU_ghost = flux_left_grad_center(Uc, Ur) * A
    left = -dFldU / Δx * Δt
    center = I2 .+ (dFldU - dFrdU_ghost - dFrdU) / Δx * Δt
    return left' * Λl + center' * Λc
end

function set_flux_jvp!(Λ, U, N, n, Δx, Δt, da::DiscreteAdjoint)
    Λ[1, n-1] = set_flux_jvp_left_boundary!(Λ[1:2, n]...,
                                             U[1:2, n-1]...,
                                             Δt, Δx, da)
    for i in 2:N-1
        Λ[i, n-1] = set_flux_jvp_interior!(Λ[i-1:i+1, n]...,
                                           U[i-1:i+1, n-1]...,
                                           Δt, Δx, da)
    end
    Λ[N, n-1] = set_flux_jvp_right_boundary!(Λ[N-1:N, n]...,
                                             U[N-1:N, n-1]...,
                                             Δt, Δx, da)
end

function add_objective_source!(Λ, objectives, U, Δt, Δx)
    indices = objectives.objective_indices
    objective = objectives.interior_objective
    Λ[indices] .+= objective_density_gradient.(objective, @view U[indices]) * Δt * Δx
end

function compute_objective_step(U, Δx, objectives::Objectives)
    indices = objectives.objective_indices
    objective = objectives.interior_objective
    Jn = sum(objective_density.(objective, @view U[indices])) * Δx
    return Jn
end


@views function solve_adjoint(Λ0, U::AverageDepthStates, objectives::Objectives, b, t, Δx, da::DiscreteAdjoint)
    U = U.U
    Λ = similar(U)

    Δt = t[end] - t[end-1]
    μ_final = compute_timestep_correction(Λ0, U[:, end-1], U[:, end], Δt)
    J_final = compute_objective_step(U[:, end-1], Δx, objectives)

    N, M = size(U)
    Λ[:, end] .= Λ0
    for n in M:-1:2
        Δt = t[n] - t[n-1]
        set_flux_jvp!(Λ, U, N, n, Δx, Δt, da)
        add_objective_source!(Λ[:, n-1], objectives, U[:, n-1], Δt, Δx)
        if n < M
            add_timestep_source!(Λ[:, n-1], Λ[:, n], U[:, n], U[:, n-1], μ_final, J_final, Δt, Δx, 0.25, objectives)
        end
    end
    return Λ
end

struct TestDiscreteAdjoint <: Adjoint end

function simulator_constructor(N, T)
    function construct_simulator(U)
        U = States{Average, Elevation}(unflatten(U))
        problem = SinFVMPrimalSWEProblem(N, U, T, reconstruction=NoReconstruction(), timestepper=ForwardEuler())
        simulator = _create_simulator(problem, zeros(eltype(eltype(U.U)), N+1))
        return simulator
    end
end


function flatten(U::AbstractVector)
    return reduce(vcat, U)
end

function flatten(U::AbstractMatrix)
    return reduce(vcat, vec(U))
end

function unflatten(u_flat)
    N = length(u_flat) ÷ 2
    [State(u_flat[2i-1:2i]) for i in 1:N]
end

function unflatten(u_flat, N)
    M = length(u_flat) ÷ (2*N)
    reshape(unflatten(u_flat), N, M)
end


function solve_adjoint(Λ0, U::AverageDepthStates, dJdU, b, t, Δx, da::TestDiscreteAdjoint)
    U0 = to_depth(States{Average, Elevation}(U.U[:,1]), -b)
    N = size(U.U, 1)
    J_flat = ForwardDiff.jacobian(flatten(U0.U)) do U
        β = zeros(eltype(eltype(U)), N+1)
        U = States{Average, Elevation}(unflatten(U))
        problem = SinFVMPrimalSWEProblem(N, U, last(t), reconstruction=NoReconstruction(), timestepper=ForwardEuler())
        U, t, x = solve_primal(problem, β)
        return flatten(U.U[:, end])
    end

    # display(J_flat')

    Λ = unflatten(J_flat' * flatten(Λ0), N)
    return Λ
end

struct StepWiseTestDiscreteAdjoint <: Adjoint end


function solve_adjoint(Λ0, U::AverageDepthStates, dJdU, b, t, Δx, da::StepWiseTestDiscreteAdjoint)
    U = to_depth(States{Average, Elevation}(U.U), -b)
    U = U.U
    Λ = similar(U)
    N, M = size(U)
    Λ[:, end] .= Λ0

    construct_simulator = simulator_constructor(N, last(t))

    J_step = DiffResults.JacobianResult(zeros(2*N))

    for n in M:-1:2
        function f(U)
            simulator = construct_simulator(U)
            max_dt = t[end]-t[n-1]
            SinFVM.perform_step!(simulator, max_dt)
            return flatten(SinFVM.current_interior_state(simulator))
        end

        ForwardDiff.jacobian!(J_step, f, flatten(U[:, n-1]))
        U_next = unflatten(J_step.value)

        d = sum(U_next - U[:, n]) do U_diff
            sum(abs2, U_diff)
        end
        # @assert d < 1e-6 "Forward step does not match stored state at time step $n≤$M. $d"
        # @assert U_next ≈ U[:, n] "Forward step does not match stored state at time step $n<$M. $d"

        Λ[:, n-1] .= unflatten(J_step.derivs[1]' * flatten(Λ[:, n]))
    end
    return Λ
end

        