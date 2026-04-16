using Pkg
using Flux
using Statistics
using Printf
using Random
using JLD2

# ==============================================================================
# PHYSICS CONSTRAINT HELPERS
# NOTE: each file below contains standalone test code that prints to stdout
#       on first include — this is a one-time side effect on model load.
# ==============================================================================

include(joinpath(@__DIR__, "..", "utils", "loss_term_functions", "position_update_loss.jl"))
include(joinpath(@__DIR__, "..", "utils", "loss_term_functions", "velocity_update_loss.jl"))
include(joinpath(@__DIR__, "..", "utils", "loss_term_functions", "yaw_update_loss.jl"))

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

# Returns per-variable MSE as a 7-element vector: [dt, x, y, yaw, vx, vy, yaw_rate]
function per_variable_loss(model, past_states, future_states)
    predictions = predict_trajectory(model, past_states, size(future_states, 1))
    diff_sq = (predictions .- future_states) .^ 2
    return vec(mean(diff_sq, dims=1))  # mean over time steps, one value per state var
end

"""
    physics_informed_loss(model, past_states, future_states;
                          λ_pos, λ_vel, λ_yaw, L)

Combined data + physics loss for the kinematic bicycle model (KBM).

State column layout (must match data pipeline):
    [dt, x, y, yaw, vx, vy, yaw_rate]
     1   2  3   4   5   6     7

Three KBM constraints applied to the predicted trajectory:

1. **Position update** (`position_update_loss`):
       x_{t+1} ≈ x_t + v_t · cos(yaw_t) · dt
       y_{t+1} ≈ y_t + v_t · sin(yaw_t) · dt
   where v_t = √(vx_t² + vy_t²).

2. **Velocity update** (`velocity_update_loss`):
       v_{t+1} ≈ v_t + a · dt
   Acceleration `a` is estimated from the last two observed steps and held
   constant across the prediction horizon (forward-Euler extrapolation).

3. **Yaw update** (`yaw_update_loss`):
       yaw_{t+1} ≈ yaw_t + (v_t / L) · tan(δ_t) · dt
   Steering angle δ_t is recovered from predicted yaw_rate:
       δ_t = atan(yaw_rate_t · L / v_t)
   ensuring the yaw and yaw_rate predictions stay mutually consistent.

# Arguments
- `λ_pos`, `λ_vel`, `λ_yaw`: weights for each physics penalty term
- `L`: vehicle wheelbase in metres (default 2.7 m, typical passenger car)
"""
function physics_informed_loss(model, past_states, future_states;
                               λ_pos=0.1, λ_vel=0.1, λ_yaw=0.1, L=2.7)
    predictions = predict_trajectory(model, past_states, size(future_states, 1))

    # Data loss
    data_loss = Flux.mse(predictions, future_states)

    # Representative timestep: mean dt over the observed window
    dt = mean(past_states[:, 1])

    # Speed magnitude at each predicted step
    v = sqrt.(predictions[:, 5].^2 .+ predictions[:, 6].^2)

    # --- Position physics loss ---
    # trajectory_position_loss expects N×4: [x, y, v, yaw]
    pos_traj = hcat(predictions[:, 2], predictions[:, 3], v, predictions[:, 4])
    pos_loss = trajectory_position_loss(pos_traj, dt)

    # --- Velocity physics loss ---
    # trajectory_velocity_loss expects N×2: [v, a]
    # Estimate acceleration from the last two observed steps; carry it forward
    # as a constant through the prediction horizon.
    v_prev  = sqrt(past_states[end-1, 5]^2 + past_states[end-1, 6]^2)
    v_last  = sqrt(past_states[end,   5]^2 + past_states[end,   6]^2)
    dt_obs  = max(past_states[end, 1], 1e-6)
    a_est   = (v_last - v_prev) / dt_obs
    vel_traj = hcat(v, fill(a_est, size(predictions, 1)))
    vel_loss = trajectory_velocity_loss(vel_traj, dt)

    # --- Yaw physics loss ---
    # trajectory_yaw_loss expects N×3: [yaw, v, delta]
    # Recover steering angle: delta = atan(yaw_rate * L / v)
    v_safe   = max.(v, 1e-3)
    delta    = atan.(predictions[:, 7] .* L ./ v_safe)
    yaw_traj = hcat(predictions[:, 4], v, delta)
    yaw_loss = trajectory_yaw_loss(yaw_traj, L, dt)

    phys_loss = λ_pos * pos_loss + λ_vel * vel_loss + λ_yaw * yaw_loss
    return data_loss + phys_loss
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
                physics_informed_loss(m, past, future)
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