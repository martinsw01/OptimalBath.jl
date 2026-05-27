export ContinuousAdjointSWE, AdjointSpec, BalancedFluxJacobianDivergence, SimpleBottomSource

abstract type FluxJacobianDivergence end
abstract type AdjointBottomSource end

struct MatchPrimalReconstruction end
struct ReconstructionDependentBottomSource end

struct BalancedFluxJacobianDivergence <: FluxJacobianDivergence end
struct SimpleBottomSource <: AdjointBottomSource end


struct ContinuousAdjointSWE{PrimalSolver, GridT, FJD, ABS, R} <: AdjointSWE
    primal::PrimalSolver
    grid::GridT
    flux_jacobian_divergence::FJD
    bottom_source::ABS
    primal_reconstruction::R
    function ContinuousAdjointSWE(primal::PrimalSWESolver;
                                  flux_jacobian_divergence=BalancedFluxJacobianDivergence(),
                                  bottom_source=ReconstructionDependentBottomSource(),
                                  primal_reconstruction=MatchPrimalReconstruction())
        grid = get_grid(primal)
        resolved_reconstruction = resolve_primal_reconstruction(primal, primal_reconstruction)
        resolved_bottom_source = resolve_bottom_source_term(resolved_reconstruction, bottom_source)
        return new{typeof(primal),
                   typeof(grid),
                   typeof(flux_jacobian_divergence),
                   typeof(resolved_bottom_source),
                   typeof(resolved_reconstruction)}(primal,
                                                    grid,
                                                    flux_jacobian_divergence,
                                                    resolved_bottom_source,
                                                    resolved_reconstruction)
    end
end

function resolve_primal_reconstruction(::PrimalSWESolver{R}, ::MatchPrimalReconstruction) where R
    return R()
end

function resolve_primal_reconstruction(_, reconstruction::Reconstruction)
    return reconstruction
end

function resolve_bottom_source_term(_, bottom_source::AdjointBottomSource)
    return bottom_source
end

function resolve_bottom_source_term(::WellBalancedReconstruction, ::ReconstructionDependentBottomSource)
    return SimpleBottomSource()
end

include("numerical_adjoint_fluxes.jl")
include("flux_jacobian_sources/flux_jacobian_sources.jl")
include("bottom_source_terms.jl")

function solve_adjoint(Λ_end, U::AverageDepthStates, objectives::Objectives, b, t, da::ContinuousAdjointSWE)
    grid = da.grid
    Δx = grid.Δx

    M = length(t)
    Λ = similar(U.U)
    time_frame(Λ, M) .= Λ_end

    U_left, U_right = reconstruction_buffers(time_frame(U.U, 1), da.primal_reconstruction)

    for n in M-1:-1:1
        Δt = t[n+1] - t[n]
        time_frame(Λ, n) .= time_frame(Λ, n+1)
        for dir in directions(grid)
            reconstruct!(U_left, U_right, time_frame(U.U, n), b, dir, grid, da.primal_reconstruction)
            add_adjoint_flux!(Λ, U_left, U_right, n, Δt, dir, grid, da)
            add_flux_spatial_derivative!(Λ, U_left, U_right, n, Δt, dir, grid, da)
            add_bottom_source!(Λ, n, Δt, b, dir, grid, da.bottom_source)
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