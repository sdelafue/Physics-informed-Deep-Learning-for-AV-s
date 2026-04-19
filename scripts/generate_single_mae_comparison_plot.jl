using JLD2
using Plots
using Statistics

# ==============================================================================
# LOAD MAE LISTS
# ==============================================================================

dd_path = joinpath(@__DIR__, "..", "exported_data", "data-driven_test_mae.jld2")
pi_path = joinpath(@__DIR__, "..", "exported_data", "Physics-informed LSTM_test_mae.jld2")

dd_mae_list = load(dd_path, "mae_list")
pi_mae_list = load(pi_path, "mae_list")

println("Loaded data-driven MAE list:       $(length(dd_mae_list)) samples")
println("Loaded physics-informed MAE list:  $(length(pi_mae_list)) samples")

length(dd_mae_list) == length(pi_mae_list) || error(
    "Sample count mismatch: data-driven=$(length(dd_mae_list)), " *
    "physics-informed=$(length(pi_mae_list))"
)

# ==============================================================================
# AVERAGE MAE ACROSS ALL 6 STATE VARIABLES PER SAMPLE
# State variable order: (x, y, yaw, vx, vy, yaw_rate)
# ==============================================================================

dd_avg = [mean(sample) for sample in dd_mae_list]
pi_avg = [mean(sample) for sample in pi_mae_list]

n_samples = length(dd_avg)

# ==============================================================================
# PLOT
# ==============================================================================

p = plot(
    1:n_samples, dd_avg;
    label       = "Data-driven",
    title       = "Overall Average MAE Comparison — Test Set",
    xlabel      = "Sample",
    ylabel      = "Mean MAE (all state variables)",
    linewidth   = 2,
    color       = :steelblue,
    legend      = :topright,
)
plot!(p,
    1:n_samples, pi_avg;
    label     = "Physics-informed",
    linewidth = 2,
    color     = :orangered,
)

# ==============================================================================
# EXPORT
# ==============================================================================

output_dir  = joinpath(@__DIR__, "..", "exported_data", "plotted_metrics", "comparison")
output_path = joinpath(output_dir, "overall_avg_mae_comparison.png")
mkpath(output_dir)
savefig(p, output_path)

println("Saved: $output_path")
