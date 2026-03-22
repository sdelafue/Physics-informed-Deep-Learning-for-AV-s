using Pkg
using Flux
using Statistics
using Random
using CSV
using DataFrames

model_filepath = "../../models/lstm_baseline.jl"
include(model_filepath)

# ==============================================================================
# EXAMPLE USAGE FOR PHASE 1: BARE MINIMUM LSTM MODEL
# ==============================================================================

function example_usage()
    state_dim = 5
    hidden_dim = 64
    n_past = 10
    n_future = 30
    n_samples = 100

    model = TrajectoryLSTM(state_dim, hidden_dim)

    # Generate dummy training data
    train_data = []
    for i in 1:n_samples
        past = randn(Float32, state_dim, n_past)
        future = randn(Float32, state_dim, n_future)
        push!(train_data, (past, future))
    end

    println("Starting training...")
    train_model!(model, train_data, epochs=50, lr=0.001)

    # Test prediction
    test_past = randn(Float32, state_dim, n_past)
    test_prediction = predict_trajectory(model, test_past, n_future)

    println("\nPrediction shape: ", size(test_prediction))
    println("Example prediction (first 5 timesteps):")
    println(test_prediction[:, 1:5])

    return model
end

# Run example
# model = example_usage()