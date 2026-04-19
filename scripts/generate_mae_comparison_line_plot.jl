using JLD2
using Plots
using Printf
using Statistics
using CSV
using DataFrames

# Report helpers (provides generate_model_comparison_mae_plot)
include(joinpath(@__DIR__, "..", "utils", "generate_plots", "report_metrics.jl"))

# ==============================================================================
# LOAD MAE LISTS
# ==============================================================================

dd_path = joinpath(@__DIR__, "..", "exported_data", "data-driven_test_mae.jld2")
pi_path = joinpath(@__DIR__, "..", "exported_data", "Physics-informed LSTM_test_mae.jld2")

dd_mae_list = load(dd_path, "mae_list")
pi_mae_list = load(pi_path, "mae_list")

println("Loaded data-driven MAE list:       $(length(dd_mae_list)) samples")
println("Loaded physics-informed MAE list:  $(length(pi_mae_list)) samples")

# ==============================================================================
# GENERATE ONE COMPARISON LINE PLOT PER STATE VARIABLE
# ==============================================================================
# Output files land in:
#   exported_data/plotted_metrics/comparison/
# and are named:
#   mae_comparison_{state_var}_model_comparison_mae.png

state_vars  = ["x", "y", "yaw", "vx", "vy", "yaw_rate"]
export_base = joinpath(@__DIR__, "..", "exported_data",
                       "plotted_metrics", "comparison", "mae_comparison")

println()
for var in state_vars
    path = generate_model_comparison_mae_plot(
        dd_mae_list, pi_mae_list, var;
        export_file_pth = export_base,
        split           = "Test Set",
    )
    println("Saved [$var]: $path")
end

println("\nAll 6 comparison plots exported.")
