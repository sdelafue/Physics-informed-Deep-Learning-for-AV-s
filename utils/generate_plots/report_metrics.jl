# Import necessary libraries below
using CSV
using DataFrames
using Plots
using Printf
using Statistics

"""
    export_to_csv(mse_prediction_list::AbstractVector{<:Tuple},
                  mae_prediction_list::AbstractVector{<:Tuple},
                  mape_prediction_list::AbstractVector{<:Tuple};
                  export_file_pth="exported_data/csv_metrics",
                  model_type="data-driven")

Exports one CSV row per sample with averaged MSE, MAE, and MAPE values for
each state variable, excluding `dt`.
"""
function export_to_csv(mse_prediction_list::AbstractVector{<:Tuple},
                       mae_prediction_list::AbstractVector{<:Tuple},
                       mape_prediction_list::AbstractVector{<:Tuple};
                       export_file_pth::AbstractString="exported_data/csv_metrics",
                       model_type::AbstractString="data-driven")
    n_samples = length(mse_prediction_list)
    length(mae_prediction_list) == n_samples || throw(ArgumentError(
        "mae_prediction_list must contain $n_samples samples. Got $(length(mae_prediction_list))."
    ))
    length(mape_prediction_list) == n_samples || throw(ArgumentError(
        "mape_prediction_list must contain $n_samples samples. Got $(length(mape_prediction_list))."
    ))

    metric_columns = [
        :x_mse, :y_mse, :yaw_mse, :vx_mse, :vy_mse, :yaw_rate_mse,
        :x_mae, :y_mae, :yaw_mae, :vx_mae, :vy_mae, :yaw_rate_mae,
        :x_mape, :y_mape, :yaw_mape, :vx_mape, :vy_mape, :yaw_rate_mape,
    ]

    metric_rows = Vector{NamedTuple{Tuple(metric_columns), NTuple{18, Float64}}}()

    for sample_idx in 1:n_samples
        mse_values = _validate_metric_tuple(mse_prediction_list[sample_idx], "mse_prediction_list", sample_idx)
        mae_values = _validate_metric_tuple(mae_prediction_list[sample_idx], "mae_prediction_list", sample_idx)
        mape_values = _validate_metric_tuple(mape_prediction_list[sample_idx], "mape_prediction_list", sample_idx)
        push!(metric_rows, NamedTuple{Tuple(metric_columns)}((
            mse_values...,
            mae_values...,
            mape_values...,
        )))
    end

    mkpath(export_file_pth)
    output_path = joinpath(export_file_pth, "$(model_type)_error.csv")
    CSV.write(output_path, DataFrame(metric_rows))

    return output_path
end

function _validate_metric_tuple(metric_values::Tuple, list_name::AbstractString, sample_idx::Integer)
    length(metric_values) == 6 || throw(ArgumentError(
        "$(list_name) sample $(sample_idx) must contain 6 values: x, y, yaw, vx, vy, yaw_rate. Got $(length(metric_values))."
    ))

    return Float64.(metric_values)
end

function plot_pred_vs_actual(actual, predicted)
    # Placeholder for plotting code
    # You can use Plots.jl or any other plotting library to visualize the results
    println("Plotting predicted vs actual trajectories...")


end

"""
    generate_mae_histogram(predicted::Matrix, actual::Matrix;
                           export_file_pth="exported_data/plotted_metrics/mae",
                           model_type="data-driven",
                           split="Test Set")

Exports a PNG bar histogram of the mean absolute error (MAE) for a single
prediction. The MAE is averaged across all prediction timestamps for each
state variable, excluding `dt`.

The `split` keyword indicates whether the sample comes from the training set
or the test set and is included in the histogram title and output filename.
"""
function generate_mae_histogram(predicted::Matrix, actual::Matrix;
                            export_file_pth::AbstractString="exported_data/plotted_metrics/mae",
                            model_type::AbstractString="data-driven",
                            split::AbstractString="Test Set")
    size(predicted) == size(actual) || throw(ArgumentError(
        "predicted and actual must have the same dimensions. Got $(size(predicted)) and $(size(actual))."
    ))
    size(predicted, 2) >= 2 || throw(ArgumentError(
        "predicted and actual must include dt plus at least one state variable column."
    ))

    state_labels = ["x", "y", "yaw", "vx", "vy", "yaw_rate"]
    state_cols = 2:size(predicted, 2)
    plotted_labels = [
        i <= length(state_labels) ? state_labels[i] : "state_$(i)"
        for i in 1:length(state_cols)
    ]

    mae_by_state = vec(mean(abs.(predicted[:, state_cols] .- actual[:, state_cols]); dims=1))

    export_dir = dirname(export_file_pth)
    isempty(export_dir) && (export_dir = ".")
    mkpath(export_dir)

    export_base, export_ext = splitext(basename(export_file_pth))
    lowercase(export_ext) == ".png" || (export_base = basename(export_file_pth))
    sample_num = _next_sample_number(export_dir, export_base)
    output_path = joinpath(export_dir, "$(export_base)_sample_$(sample_num)_$(model_type)_mae.png")

    p = bar(
        plotted_labels,
        mae_by_state;
        title="sample $(sample_num) $(model_type) MAE ($(split))",
        xlabel="State variable",
        ylabel="Mean absolute error",
        legend=false,
        xrotation=30,
        color=:steelblue,
    )

    for (i, v) in enumerate(mae_by_state)
        annotate!(p, i, v, text(@sprintf("%.4f", v), :center, :bottom, 8))
    end

    savefig(p, output_path)
    return output_path
end

"""
    generate_grand_average_mae_histogram(mae_prediction_list::AbstractVector{<:Tuple};
                                         export_file_pth="exported_data/plotted_metrics/mae",
                                         model_type="data-driven",
                                         split="Test Set")

Computes the grand average MAE across all samples in `mae_prediction_list` and
exports a PNG bar histogram of the result.

Each element of `mae_prediction_list` must be a 6-element tuple of `Float64`
values in the order `(x, y, yaw, vx, vy, yaw_rate)`, matching the format
produced by `export_to_csv`. The `dt` variable is intentionally excluded.

Prints the per-variable grand average MAE and the overall mean across all
state variables to stdout. Returns the path of the saved PNG.
"""
function generate_grand_average_mae_histogram(
    mae_prediction_list::AbstractVector{<:Tuple};
    export_file_pth::AbstractString="exported_data/plotted_metrics/mae",
    model_type::AbstractString="data-driven",
    split::AbstractString="Test Set",
)
    isempty(mae_prediction_list) && throw(ArgumentError(
        "mae_prediction_list must contain at least one sample."
    ))

    n_samples = length(mae_prediction_list)
    validated = [_validate_metric_tuple(mae_prediction_list[i], "mae_prediction_list", i)
                 for i in 1:n_samples]

    # Average each state variable's MAE across all samples
    grand_avg = vec(mean(stack(validated, dims=1), dims=1))

    state_labels = ["x", "y", "yaw", "vx", "vy", "yaw_rate"]

    # Print per-variable and overall averages
    println("Grand Average MAE — $(model_type) model ($(split)), n=$(n_samples) samples:")
    for (label, v) in zip(state_labels, grand_avg)
        @printf("  %-10s %.4f\n", label * ":", v)
    end
    @printf("  %-10s %.4f\n", "overall:", mean(grand_avg))

    export_dir = dirname(export_file_pth)
    isempty(export_dir) && (export_dir = ".")
    mkpath(export_dir)

    export_base, export_ext = splitext(basename(export_file_pth))
    lowercase(export_ext) == ".png" || (export_base = basename(export_file_pth))
    output_path = joinpath(export_dir, "$(export_base)_$(model_type)_grand_avg_mae.png")

    p = bar(
        state_labels,
        grand_avg;
        title="Grand Average MAE — $(model_type) ($(split), n=$(n_samples))",
        xlabel="State variable",
        ylabel="Mean absolute error",
        legend=false,
        xrotation=30,
        color=:steelblue,
    )

    for (i, v) in enumerate(grand_avg)
        annotate!(p, i, v, text(@sprintf("%.4f", v), :center, :bottom, 8))
    end

    savefig(p, output_path)
    return output_path
end

"""
    generate_model_comparison_mae_plot(dd_mae_list, pi_mae_list, state_var;
                                       export_file_pth, split)

Exports a PNG line plot comparing per-sample MAE for a single state variable
between the data-driven model and the physics-informed model across a split.

Each element of `dd_mae_list` and `pi_mae_list` must be a 6-element tuple in
the order `(x, y, yaw, vx, vy, yaw_rate)`, matching the format used by
`export_to_csv` and `generate_grand_average_mae_histogram`. Both lists must
have the same length.

`state_var` must be one of: `"x"`, `"y"`, `"yaw"`, `"vx"`, `"vy"`,
`"yaw_rate"`.

The x-axis is the sample index; the y-axis is the MAE for `state_var`. The
two lines are labelled "Data-driven" and "Physics-informed". Returns the path
of the saved PNG.
"""
function generate_model_comparison_mae_plot(
    dd_mae_list::AbstractVector{<:Tuple},
    pi_mae_list::AbstractVector{<:Tuple},
    state_var::AbstractString;
    export_file_pth::AbstractString="exported_data/plotted_metrics/comparison",
    split::AbstractString="Test Set",
)
    state_labels = ["x", "y", "yaw", "vx", "vy", "yaw_rate"]

    var_idx = findfirst(==(state_var), state_labels)
    var_idx === nothing && throw(ArgumentError(
        "state_var must be one of $(state_labels). Got \"$(state_var)\"."
    ))
    isempty(dd_mae_list) && throw(ArgumentError(
        "dd_mae_list must contain at least one sample."
    ))
    isempty(pi_mae_list) && throw(ArgumentError(
        "pi_mae_list must contain at least one sample."
    ))
    length(dd_mae_list) == length(pi_mae_list) || throw(ArgumentError(
        "dd_mae_list and pi_mae_list must have the same length. " *
        "Got $(length(dd_mae_list)) and $(length(pi_mae_list))."
    ))

    n_samples = length(dd_mae_list)

    dd_mae = [_validate_metric_tuple(dd_mae_list[i], "dd_mae_list", i)[var_idx]
              for i in 1:n_samples]
    pi_mae = [_validate_metric_tuple(pi_mae_list[i], "pi_mae_list", i)[var_idx]
              for i in 1:n_samples]

    export_dir = dirname(export_file_pth)
    isempty(export_dir) && (export_dir = ".")
    mkpath(export_dir)

    export_base, export_ext = splitext(basename(export_file_pth))
    lowercase(export_ext) == ".png" || (export_base = basename(export_file_pth))
    output_path = joinpath(export_dir, "$(export_base)_$(state_var)_model_comparison_mae.png")

    sample_indices = 1:n_samples

    p = plot(
        sample_indices, dd_mae;
        label="Data-driven",
        title="MAE Comparison ($(state_var)) — $(split)",
        xlabel="Sample",
        ylabel="MAE ($(state_var))",
        linewidth=2,
        color=:steelblue,
        legend=:topright,
    )
    plot!(p,
        sample_indices, pi_mae;
        label="Physics-informed",
        linewidth=2,
        color=:orangered,
    )

    savefig(p, output_path)
    return output_path
end

function _next_sample_number(export_dir::AbstractString, export_base::AbstractString)
    # Match any filename of the form "{export_base}_sample_{N}*.png"
    # Using regex so that any suffix after the number (e.g. "_data-driven_mae")
    # does not interfere with parsing the sample index.
    pattern = Regex("^$(export_base)_sample_(\\d+)")
    sample_nums = Int[]

    if isdir(export_dir)
        for file_name in readdir(export_dir)
            m = match(pattern, file_name)
            m === nothing && continue
            push!(sample_nums, parse(Int, m.captures[1]))
        end
    end

    return isempty(sample_nums) ? 1 : maximum(sample_nums) + 1
end
