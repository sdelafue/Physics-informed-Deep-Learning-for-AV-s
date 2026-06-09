using Pkg
using Flux
using JLD2
using Statistics
using Random
using CSV
using DataFrames
using Printf

# Includes are managed by the entry point (test.jl).
# This file assumes TrajectoryLSTM, train_model!, predict_trajectory,
# and the data loading pipeline are already defined in the calling scope.

# ==============================================================================
# EXAMPLE USAGE FOR PHASE 1: BARE MINIMUM LSTM MODEL (DUMMY DATA; DATA DRIVEN ONLY)
# ==============================================================================

function example_usage()
    state_dim = 7  # [dt, x, y, yaw, vx, vy, yaw_rate]
    hidden_dim = 64
    n_past = 10
    n_future = 30
    n_samples = 100

    model = TrajectoryLSTM(state_dim, hidden_dim)

    # Generate dummy training data
    train_data = []
    for i in 1:n_samples
        past = randn(Float32, n_past, state_dim)
        future = randn(Float32, n_future, state_dim)
        push!(train_data, (past, future))
    end

    println("Starting training...")
    train_model!(model, train_data, epochs=50, lr=0.001)

    # Test prediction
    test_past = randn(Float32, n_past, state_dim)
    test_prediction = predict_trajectory(model, test_past, n_future)

    println("\nPrediction shape: ", size(test_prediction))
    println("Example prediction (first 5 timesteps):")
    println(test_prediction[1:5, :])

    return model
end

# Run example
# model = example_usage()

# ==============================================================================
# REAL DATA TRAINING AND PREDICTION EXPORT
# ==============================================================================

function train_on_real_data(model_type::String;
                              epochs::Int=100, lr::Float64=0.001,
                              hidden_dim::Int=64,
                              report_dir::String="exported_data")
    state_dim = 7   # [dt, x, y, yaw, vx, vy, yaw_rate]

    # ------------------------------------------------------------------
    # Step 1: Load real data via the preprocessing pipeline
    # ------------------------------------------------------------------
    println("Loading and preprocessing data...")
    raw_samples = main()   # returns shuffled Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}
    samples = [(Float32.(p), Float32.(f)) for (p, f) in raw_samples]
    println("Total samples loaded: $(length(samples))")

    # ------------------------------------------------------------------
    # Step 2: 80/20 train/test split
    # ------------------------------------------------------------------
    n_train = floor(Int, 0.8 * length(samples))
    train_data = samples[1:n_train]
    test_data  = samples[n_train+1:end]
    println("Train samples: $n_train  |  Test samples: $(length(test_data))")

    # ------------------------------------------------------------------
    # Step 3: Build model and train
    # ------------------------------------------------------------------
    model = TrajectoryLSTM(state_dim, hidden_dim)
    println("\nStarting training on real data...")
    train_model!(model, train_data, epochs=epochs, lr=lr)
    save_model(model, joinpath(report_dir, "$(model_type)_model"))
    println("Model saved to: $(joinpath(report_dir, "$(model_type)_model.jld2"))")

    # ------------------------------------------------------------------
    # Step 4: Run predictions on test split and accumulate per-sample metrics
    # ------------------------------------------------------------------
    state_cols      = 2:state_dim   # columns 2-7 exclude dt
    # The report helpers split export_file_pth into dirname (output directory)
    # and basename (filename prefix), so we include a "mae" leaf to get the
    # correct subdirectory structure.
    per_sample_base = joinpath(report_dir, "plotted_metrics", "per_sample", "mae")
    grand_avg_base  = joinpath(report_dir, "plotted_metrics", "grand_averages", "mae")
    csv_dir         = joinpath(report_dir, "csv_metrics")

    mse_list  = Tuple[]
    mae_list  = Tuple[]
    mape_list = Tuple[]

    for (past, future) in test_data
        n_future_steps = size(future, 1)
        prediction     = predict_trajectory(model, past, n_future_steps)

        pred_f64   = Float64.(prediction)
        actual_f64 = Float64.(future)

        mse_vals  = vec(mean(
            (pred_f64[:, state_cols] .- actual_f64[:, state_cols]).^2; dims=1))
        mae_vals  = vec(mean(
            abs.(pred_f64[:, state_cols] .- actual_f64[:, state_cols]); dims=1))
        mape_vals = vec(mean(
            abs.((pred_f64[:, state_cols] .- actual_f64[:, state_cols]) ./
                 (abs.(actual_f64[:, state_cols]) .+ 1e-8)) .* 100; dims=1))

        push!(mse_list,  Tuple(mse_vals))
        push!(mae_list,  Tuple(mae_vals))
        push!(mape_list, Tuple(mape_vals))

        generate_mae_histogram(pred_f64, actual_f64;
                               export_file_pth=per_sample_base,
                               model_type=model_type,
                               split="Test Set")
    end

    # ------------------------------------------------------------------
    # Step 5: Persist MAE list for cross-model comparison plots
    # ------------------------------------------------------------------
    mae_jld2_path = joinpath(report_dir, "$(model_type)_test_mae.jld2")
    jldsave(mae_jld2_path; mae_list)
    println("Test MAE list saved to: $mae_jld2_path")

    # ------------------------------------------------------------------
    # Step 6: Grand-average MAE histogram across the full test split
    # ------------------------------------------------------------------
    generate_grand_average_mae_histogram(mae_list;
                                         export_file_pth=grand_avg_base,
                                         model_type=model_type,
                                         split="Test Set")

    # ------------------------------------------------------------------
    # Step 7: Export per-sample MSE / MAE / MAPE to CSV
    # ------------------------------------------------------------------
    export_to_csv(mse_list, mae_list, mape_list;
                  export_file_pth=csv_dir,
                  model_type=model_type)

    println("Reports exported to: $report_dir")
    return model
end