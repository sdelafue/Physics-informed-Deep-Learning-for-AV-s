# sandbox.jl
using Pkg
using Flux
using Statistics
using Printf
using Random
using JLD2

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
    h = zeros(eltype(past_states), hidden_dim)
    c = zeros(eltype(past_states), hidden_dim)


    # Encode the past trajectory
    for t in 1:size(past_states, 1)
        _, (h, c) = model.encoder(past_states[t, :], (h, c))
    end

    # Autoregressively predict future using Zygote-friendly Flux.Recur pattern
    # Use foldl to accumulate predictions without in-place mutation
    current_state = past_states[end, :]
    init = (current_state, h, c, similar(past_states, 0, size(past_states, 2)))

    result = foldl(1:n_future; init=init) do (cur, h_acc, c_acc, preds), _
        output, (h_new, c_new) = model.encoder(cur, (h_acc, c_acc))
        next_state = model.decoder(output)
        (next_state, h_new, c_new, vcat(preds, next_state'))
    end

    return result[4]
end

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

function baseline_loss(model, past_states, future_states)
    predictions = predict_trajectory(model, past_states, size(future_states, 1))
    return Flux.mse(predictions, future_states)
end

# Returns per-variable MSE as a 5-element vector: [x, y, v, a, yaw]
function per_variable_loss(model, past_states, future_states)
    predictions = predict_trajectory(model, past_states, size(future_states, 1))
    diff_sq = (predictions .- future_states) .^ 2
    return vec(mean(diff_sq, dims=1))  # mean over time steps, one value per state var
end

"""
Physics-informed loss (pending full implementation).
"""
function physics_informed_loss(model, past_states, future_states, λ=0.1)
    predictions = predict_trajectory(model, past_states, size(future_states, 1))

    # Data loss
    data_loss = Flux.mse(predictions, future_states)
    
    # Physics loss (placeholder - you'll add your constraint functions)
    # phys_loss = position_constraint(predictions) + 
    #             velocity_constraint(predictions) + 
    #             yaw_constraint(predictions)
    
    # return data_loss + λ * phys_loss
    
    return data_loss  # For now, just data loss
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

            # Accumulate per-variable MSE (no gradients needed)
            state_dim = size(first(train_data)[2], 2)
            var_losses = zeros(Float64, state_dim)
            for (past, future) in train_data
                var_losses .+= per_variable_loss(model, past, future)
            end
            var_losses ./= length(train_data)

            for i in 1:state_dim
                @printf("  var_%d = %.6f\n", i, var_losses[i])
            end
        end
    end
end


"""
    save_model(model::TrajectoryLSTM, filename::String="trajectory_lstm.jld2")

Save the parameters of a `TrajectoryLSTM` model to a `.jld2` file in the
current working directory. The file stores the model state (all learnable
parameters) under the key `"model_state"` and can be restored with
`Flux.loadmodel!`.
"""
function save_model(model::TrajectoryLSTM, filename::String="trajectory_lstm.jld2")
    if !endswith(filename, ".jld2")
        filename *= ".jld2"
    end
    path = joinpath(pwd(), filename)
    model_state = Flux.state(model)
    jldsave(path; model_state)
    println("Model saved to: $path")
    return path
end