# sandbox.jl
using Pkg
using Flux
using Statistics
using Random
using CSV
using DataFrames

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

struct TrajectoryLSTM
    encoder::Flux.LSTMCell
    decoder::Chain
end

Flux.@layer TrajectoryLSTM

function TrajectoryLSTM(state_dim::Int, hidden_dim::Int=64)
    encoder = Flux.LSTMCell(state_dim => hidden_dim)
    decoder = Chain(
        Dense(hidden_dim => 32, relu),
        Dense(32 => state_dim)
    )
    return TrajectoryLSTM(encoder, decoder)
end



# ==============================================================================
# MULTI-STEP PREDICTION (FIXED ARGUMENT ORDER)
# ==============================================================================

function predict_trajectory(model::TrajectoryLSTM, past_states, n_future::Int)
    hidden_dim = size(model.encoder.Wi, 1) ÷ 4

    # Initialize hidden and cell states as zeros
    h = zeros(Float32, hidden_dim)
    c = zeros(Float32, hidden_dim)

    # Encode the past trajectory
    for t in 1:size(past_states, 2)
        _, (h, c) = model.encoder(past_states[:, t], (h, c))
    end

    # Autoregressively predict future using Zygote-friendly Flux.Recur pattern
    # Use foldl to accumulate predictions without in-place mutation
    current_state = past_states[:, end]
    init = (current_state, h, c, similar(past_states, size(past_states, 1), 0))

    result = foldl(1:n_future; init=init) do (cur, h_acc, c_acc, preds), _
        output, (h_new, c_new) = model.encoder(cur, (h_acc, c_acc))
        next_state = model.decoder(output)
        (next_state, h_new, c_new, hcat(preds, next_state))
    end

    return result[4]
end

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

function baseline_loss(model, past_states, future_states)
    predictions = predict_trajectory(model, past_states, size(future_states, 2))
    return Flux.mse(predictions, future_states)
end

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

function train_model!(model, train_data; epochs::Int=100, lr::Float64=0.001)
    opt_state = Flux.setup(Flux.Adam(lr), model)

    for epoch in 1:epochs
        total_loss = 0.0

        for (past, future) in train_data
            loss, grads = Flux.withgradient(model) do m
                baseline_loss(m, past, future)
            end

            Flux.update!(opt_state, model, grads[1])
            total_loss += loss
        end

        avg_loss = total_loss / length(train_data)

        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $(round(avg_loss, digits=6))")
        end
    end
end


"""
Load trajectory sequences from your preprocessed CSV files.

Assumes CSV has columns: sample_token, timestamp, x, y, v, a, yaw
And data is sorted by sample_token, then timestamp.
"""
function load_trajectory_data(csv_path::String, n_past::Int=10, n_future::Int=30)
    df = CSV.read(csv_path, DataFrame)
    
    # Group by sample_token (each represents one scene/sequence)
    grouped = groupby(df, :sample_token)
    
    train_data = []
    
    for group in grouped
        # Extract state variables
        states = Matrix(group[:, [:x, :y, :v, :a, :yaw]])'  # Transpose to (5, T)
        
        n_timesteps = size(states, 2)
        
        # Create sliding windows
        for i in 1:(n_timesteps - n_past - n_future + 1)
            past = states[:, i:(i + n_past - 1)]
            future = states[:, (i + n_past):(i + n_past + n_future - 1)]
            
            push!(train_data, (Float32.(past), Float32.(future)))
        end
    end
    
    return train_data
end

# Usage:
# train_data = load_trajectory_data("path/to/your/trajectories.csv")
# model = TrajectoryLSTM(5, 64)
# train_model!(model, train_data, epochs=100, lr=0.001)

"""
Physics-informed loss (pending full implementation).
"""
function physics_informed_loss(model, past_states, future_states, λ=0.1)
    predictions = predict_trajectory(model, past_states, size(future_states, 2))
    
    # Data loss
    data_loss = Flux.mse(predictions, future_states)
    
    # Physics loss (placeholder - you'll add your constraint functions)
    # phys_loss = position_constraint(predictions) + 
    #             velocity_constraint(predictions) + 
    #             yaw_constraint(predictions)
    
    # return data_loss + λ * phys_loss
    
    return data_loss  # For now, just data loss
end