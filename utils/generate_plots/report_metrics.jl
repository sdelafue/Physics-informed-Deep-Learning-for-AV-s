# Import necessary libraries below
using CSV
using DataFrames
using Plots
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
    generate_histogram(predicted::Matrix, actual::Matrix;
                       export_file_pth="exported_data/plotted_metrics/mae",
                       model_type="data-driven")

Exports a PNG bar histogram of the mean absolute error (MAE) for a single
prediction. The MAE is averaged across all prediction timestamps for each
state variable, excluding `dt`.
"""
function generate_mae_histogram(predicted::Matrix, actual::Matrix;
                            export_file_pth::AbstractString="exported_data/plotted_metrics/mae",
                            model_type::AbstractString="data-driven")
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
        title="sample $(sample_num) $(model_type) MAE",
        xlabel="State variable",
        ylabel="Mean absolute error",
        legend=false,
        xrotation=30,
        color=:steelblue,
    )

    savefig(p, output_path)
    return output_path
end

function _next_sample_number(export_dir::AbstractString, export_base::AbstractString)
    prefix = "$(export_base)_sample_"
    suffix = ".png"
    sample_nums = Int[]

    if isdir(export_dir)
        for file_name in readdir(export_dir)
            startswith(file_name, prefix) || continue
            endswith(file_name, suffix) || continue

            sample_text = file_name[length(prefix)+1:end-length(suffix)]
            sample_num = tryparse(Int, sample_text)
            sample_num === nothing && continue
            push!(sample_nums, sample_num)
        end
    end

    return isempty(sample_nums) ? 1 : maximum(sample_nums) + 1
end
