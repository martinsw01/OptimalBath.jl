export ContinuousAdjointSWE

abstract type FluxJacobianDivergence end
abstract type AdjointBottomSource end

struct BalancedFluxJacobianDivergence <: FluxJacobianDivergence end
struct SimpleBottomSource <: AdjointBottomSource end

struct AdjointSpec{FJD, ABS, PrimalReconstruction}
    flux_jacobian_divergence::FJD
    bottom_source::ABS
    primal_reconstruction::PrimalReconstruction
    function AdjointSpec(;fjd::FluxJacobianDivergence=BalancedFluxJacobianDivergence(),
                         bs::AdjointBottomSource=SimpleBottomSource(),
                         reconstruction::Reconstruction=WellBalancedNoReconstruction())
        FJD = typeof(fjd)
        ABS = typeof(bs)
        PrimalReconstruction = typeof(reconstruction)
        return new{FJD, ABS, PrimalReconstruction}(fjd, bs, reconstruction)
    end
end


struct ContinuousAdjointSWE{PrimalSolver, GridT, FJD, ABS, R} <: AdjointSWE
    primal::PrimalSolver
    grid::GridT
    flux_jacobian_divergence::FJD
    bottom_source::ABS
    primal_reconstruction::R
    function ContinuousAdjointSWE(primal::PrimalSWESolver, adjoint_spec::AdjointSpec=AdjointSpec())
        grid = get_grid(primal)
        fjd = adjoint_spec.flux_jacobian_divergence
        bs = adjoint_spec.bottom_source
        reconstruction = adjoint_spec.primal_reconstruction
        return new{typeof(primal), typeof(grid), typeof(fjd), typeof(bs), typeof(reconstruction)}(primal, grid, fjd, bs, reconstruction)
    end
end

include("numerical_adjoint_fluxes.jl")
include("flux_jacobian_sources.jl")
include("bottom_source_terms.jl")

function solve_adjoint(Λ_end, U::AverageDepthStates, objectives::Objectives, b, t, da::ContinuousAdjointSWE)
    grid = da.grid
    Δx = grid.Δx

    M = length(t)
    Λ = similar(U.U)
    time_frame(Λ, M) .= Λ_end

    U_left = similar(time_frame(U.U, 1))
    U_right = similar(U_left)

    for n in M-1:-1:1
        Δt = t[n+1] - t[n]
        time_frame(Λ, n) .= time_frame(Λ, n+1)
        for dir in directions(grid)
            reconstruct!(U_left, U_right, time_frame(U.U, n), b, dir, grid, da.primal_reconstruction)
            add_adjoint_flux!(Λ, U_left, U_right, n, Δt, dir, grid, da)
            add_flux_spatial_derivative!(Λ, U_left, U_right, n, Δt, dir, grid, da)
            add_bottom_source!(Λ, n, Δt, b, dir, grid, da)
        end
        add_objective_source!(Λ, U.U, n, Δt, Δx, objectives, da)
    end
    return Λ
end

function add_objective_source!(Λ, U, n, Δt, Δx, objectives, ::ContinuousAdjointSWE)
    OptimalBath.add_objective_source!(time_frame(Λ, n), time_frame(U, n), Δt, Δx, objectives)
end


function primal_solver(da::ContinuousAdjointSWE)
    return da.primal
end