export PlotBackend, PlotsBackend, MakieBackend
export animate_solution, animate_adjoint_solution, animate_optimization
export plot_gradient, plot_objective, plot_objective_and_gradient_norm


abstract type PlotBackend end
struct PlotsBackend <: PlotBackend end
struct MakieBackend <: PlotBackend end

function animate_solution end
function animate_adjoint_solution end
function plot_gradient end
function animate_optimization end
function plot_objective_and_gradient_norm end
