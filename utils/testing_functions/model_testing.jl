using Pkg
using Flux
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

function train_on_real_data(; epochs::Int=100, lr::Float64=0.001,
                              hidden_dim::Int=64,
                              output_path::String="predictions.txt")
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

    # ------------------------------------------------------------------
    # Step 4: Run predictions on test split and export to text file
    # ------------------------------------------------------------------
    feature_names = ["dt", "x", "y", "yaw", "vx", "vy", "yaw_rate"]
    col_width = 14

    open(output_path, "w") do f
        println(f, "TRAJECTORY LSTM — TEST SET PREDICTIONS")
        println(f, "Model: state_dim=$state_dim  hidden_dim=$hidden_dim  epochs=$epochs  lr=$lr")
        println(f, "Test samples: $(length(test_data))")
        println(f, "="^(col_width * state_dim + 20))

        for (i, (past, future)) in enumerate(test_data)
            n_future  = size(future, 1)

            prediction = predict_trajectory(model, past, n_future)

            # ------ sample header ------
            println(f, "\nSample $i")
            println(f, "-"^(col_width * state_dim + 20))

            # ------ column headers ------
            header = join(lpad.(feature_names, col_width), "")
            println(f, lpad("timestep", 10) * header)
            println(f, "-"^(col_width * state_dim + 10))

            # ------ one line per predicted timestep ------
            for t in 1:n_future
                row_vals = join([@sprintf("%14.6f", prediction[t, j]) for j in 1:state_dim], "")
                println(f, lpad(string(t), 10) * row_vals)
            end
        end
    end

    println("\nPredictions written to: $output_path")
    return model
end