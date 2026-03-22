"""
Physics-Informed Loss Function: Velocity Update (Kinematic Bicycle Model)
Senior Design Project - Autonomous Vehicle Trajectory Prediction

This module implements the velocity update constraint from the Kinematic Bicycle Model
as a loss function for physics-informed deep learning.
"""

using Printf

"""
    velocity_update_loss(v_t, v_t_next, a_t, dt)

Compute the physics-informed loss for velocity updates based on the Kinematic Bicycle Model.

The physics equation enforced is:
    v_{t+1} = v_t + a_t * dt

# Arguments
- `v_t::Float64`: Current velocity (meters/second)
- `v_t_next::Float64`: Next timestep velocity (meters/second)
- `a_t::Float64`: Current acceleration (meters/second²)
- `dt::Float64`: Time step duration (seconds)

# Returns
- `Float64`: Squared residual loss

# Physics Interpretation
The loss measures how much the observed velocity change deviates from what
simple kinematic acceleration predicts. A loss of 0 means the velocity change
perfectly obeys v = v₀ + at.

# Example
```julia
loss = velocity_update_loss(10.0, 10.2, 2.0, 0.1)
```
"""
function velocity_update_loss(v_t, v_t_next, a_t, dt)
    # Predict next velocity based on kinematic equation
    v_pred = v_t + a_t * dt
    
    # Calculate residual (difference between observed and physics-predicted)
    residual = v_t_next - v_pred
    
    # Return squared residual as loss
    return residual^2
end


"""
    trajectory_velocity_loss(trajectory_data, dt)

Compute mean velocity update loss over an entire trajectory sequence.

# Arguments
- `trajectory_data::Matrix{Float64}`: N×2 matrix where each row is [v, a]
- `dt::Float64`: Time step duration (seconds)

# Returns
- `Float64`: Mean physics loss averaged over all trajectory steps

# Example
```julia
# Create a 10-step trajectory
trajectory = [velocities accelerations]
loss = trajectory_velocity_loss(trajectory, 0.1)
```
"""
function trajectory_velocity_loss(trajectory_data, dt)
    total_loss = 0.0
    n_steps = size(trajectory_data, 1) - 1
    
    for t in 1:n_steps
        # Extract current velocity and acceleration
        v_t, a_t = trajectory_data[t, :]
        v_t_next = trajectory_data[t+1, 1]
        
        # Accumulate loss for this timestep
        loss_t = velocity_update_loss(v_t, v_t_next, a_t, dt)
        total_loss += loss_t
    end
    
    return total_loss / n_steps
end


# ============================================================================
# TEST CASES
# ============================================================================

println("="^70)
println("Testing Physics-Informed Velocity Update Loss Function")
println("="^70)
println()

# Test 1: Constant velocity (zero acceleration, physics-compliant)
println("Test 1: Constant Velocity (Zero Acceleration)")
println("-"^70)
v_t = 10.0                     # Current velocity: 10 m/s
a_t = 0.0                      # No acceleration
dt = 0.1                       # Time step: 0.1 seconds
v_t_next = 10.0                # Velocity should remain 10 m/s

loss = velocity_update_loss(v_t, v_t_next, a_t, dt)
@printf("Initial velocity: %.2f m/s\n", v_t)
@printf("Acceleration: %.2f m/s², dt: %.2f s\n", a_t, dt)
@printf("Next velocity: %.2f m/s\n", v_t_next)
@printf("Physics Loss: %.6f (should be ~0.0)\n", loss)
println()

# Test 2: Positive acceleration (physics-compliant)
println("Test 2: Positive Acceleration (Speeding Up)")
println("-"^70)
v_t = 10.0                     # Current velocity: 10 m/s
a_t = 2.0                      # Accelerating at 2 m/s²
dt = 0.1                       # Time step: 0.1 seconds
v_t_next = 10.2                # After 0.1s: 10 + 2*0.1 = 10.2 m/s

loss = velocity_update_loss(v_t, v_t_next, a_t, dt)
@printf("Initial velocity: %.2f m/s\n", v_t)
@printf("Acceleration: %.2f m/s², dt: %.2f s\n", a_t, dt)
@printf("Next velocity: %.2f m/s\n", v_t_next)
@printf("Physics Loss: %.6f (should be ~0.0)\n", loss)
println()

# Test 3: Negative acceleration/deceleration (physics-compliant)
println("Test 3: Negative Acceleration (Braking)")
println("-"^70)
v_t = 15.0                     # Current velocity: 15 m/s
a_t = -3.0                     # Decelerating at -3 m/s²
dt = 0.1                       # Time step: 0.1 seconds
v_t_next = 14.7                # After 0.1s: 15 + (-3)*0.1 = 14.7 m/s

loss = velocity_update_loss(v_t, v_t_next, a_t, dt)
@printf("Initial velocity: %.2f m/s\n", v_t)
@printf("Acceleration: %.2f m/s², dt: %.2f s\n", a_t, dt)
@printf("Next velocity: %.2f m/s\n", v_t_next)
@printf("Physics Loss: %.6f (should be ~0.0)\n", loss)
println()

# Test 4: Starting from rest (physics-compliant)
println("Test 4: Starting from Rest")
println("-"^70)
v_t = 0.0                      # Starting from rest
a_t = 5.0                      # Accelerating at 5 m/s²
dt = 0.1                       # Time step: 0.1 seconds
v_t_next = 0.5                 # After 0.1s: 0 + 5*0.1 = 0.5 m/s

loss = velocity_update_loss(v_t, v_t_next, a_t, dt)
@printf("Initial velocity: %.2f m/s\n", v_t)
@printf("Acceleration: %.2f m/s², dt: %.2f s\n", a_t, dt)
@printf("Next velocity: %.2f m/s\n", v_t_next)
@printf("Physics Loss: %.6f (should be ~0.0)\n", loss)
println()

# Test 5: Physics violation (incorrect velocity change)
println("Test 5: Physics Violation (Incorrect Velocity)")
println("-"^70)
v_t = 10.0                     # Current velocity: 10 m/s
a_t = 2.0                      # Accelerating at 2 m/s²
dt = 0.1                       # Time step: 0.1 seconds
v_t_next = 12.0                # WRONG! Should be 10.2 m/s, not 12.0 m/s

loss = velocity_update_loss(v_t, v_t_next, a_t, dt)
@printf("Initial velocity: %.2f m/s\n", v_t)
@printf("Acceleration: %.2f m/s², dt: %.2f s\n", a_t, dt)
@printf("Next velocity: %.2f m/s [INCORRECT - should be 10.2 m/s]\n", v_t_next)
@printf("Physics Loss: %.6f (should be HIGH due to violation)\n", loss)
println()

# Test 6: Multi-step trajectory with varying acceleration
println("Test 6: Multi-Step Trajectory (Acceleration → Constant → Braking)")
println("-"^70)
println("Testing a 7-step trajectory with different acceleration phases...")

dt = 0.1
# Create trajectory: accelerate, cruise, then brake
trajectory = zeros(8, 2)  # 8 timesteps, 2 variables [v, a]

# Phase 1: Accelerating from rest (steps 0-2)
trajectory[1, :] = [0.0, 2.0]    # Start at rest, accel = 2 m/s²
trajectory[2, :] = [0.2, 2.0]    # v = 0 + 2*0.1 = 0.2
trajectory[3, :] = [0.4, 2.0]    # v = 0.2 + 2*0.1 = 0.4

# Phase 2: Constant velocity (steps 3-5)
trajectory[4, :] = [0.6, 0.0]    # v = 0.4 + 2*0.1 = 0.6, then coast
trajectory[5, :] = [0.6, 0.0]    # v = 0.6 + 0*0.1 = 0.6
trajectory[6, :] = [0.6, 0.0]    # v = 0.6 + 0*0.1 = 0.6

# Phase 3: Braking (steps 6-7)
trajectory[7, :] = [0.6, -1.0]   # Start braking at -1 m/s²
trajectory[8, :] = [0.5, -1.0]   # v = 0.6 + (-1)*0.1 = 0.5

println("\nTrajectory data:")
println("Step | v (m/s) | a (m/s²) | Phase")
println("-----|---------|----------|------------------")
for i in 1:8
    phase = i <= 3 ? "Accelerating" : (i <= 6 ? "Coasting" : "Braking")
    @printf("%4d | %.2f    | %.2f     | %s\n", i-1, trajectory[i,1], trajectory[i,2], phase)
end

traj_loss = trajectory_velocity_loss(trajectory, dt)
@printf("\nMean Trajectory Loss: %.6f (should be ~0.0)\n", traj_loss)
println()

println("="^70)
println("All tests complete!")
println("="^70)