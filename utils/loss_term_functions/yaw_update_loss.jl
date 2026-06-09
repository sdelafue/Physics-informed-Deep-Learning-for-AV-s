"""
Physics-Informed Loss Function: Yaw Update (Kinematic Bicycle Model)
Senior Design Project - Autonomous Vehicle Trajectory Prediction

This module implements the yaw (heading) update constraint from the Kinematic Bicycle Model
as a loss function for physics-informed deep learning.
"""

using Printf

"""
    yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)

Compute the physics-informed loss for yaw (heading) updates based on the Kinematic Bicycle Model.

The physics equation enforced is:
    yaw_{t+1} = yaw_t + (v_t / L) * tan(delta_t) * dt

# Arguments
- `yaw_t::Float64`: Current yaw angle/heading (radians)
- `yaw_t_next::Float64`: Next timestep yaw angle (radians)
- `v_t::Float64`: Current velocity (meters/second)
- `delta_t::Float64`: Current steering angle (radians)
- `L::Float64`: Wheelbase length of the vehicle (meters)
- `dt::Float64`: Time step duration (seconds)

# Returns
- `Float64`: Squared residual loss

# Physics Interpretation
The loss measures how much the observed yaw change deviates from what the kinematic
bicycle model predicts based on velocity, steering angle, and wheelbase geometry.
A loss of 0 means the heading change perfectly obeys the bicycle model kinematics.

# Note on Steering Angle
- delta = 0: driving straight (no yaw change)
- delta > 0: turning left (counter-clockwise, yaw increases)
- delta < 0: turning right (clockwise, yaw decreases)

# Example
```julia
loss = yaw_update_loss(0.0, 0.05, 10.0, 0.1, 2.5, 0.1)
```
"""
function yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
    # Predict next yaw based on kinematic bicycle model
    yaw_rate = (v_t / L) * tan(delta_t)
    yaw_pred = yaw_t + yaw_rate * dt
    
    # Calculate residual (difference between observed and physics-predicted)
    residual = yaw_t_next - yaw_pred
    
    # Return squared residual as loss
    return residual^2
end


"""
    trajectory_yaw_loss(trajectory_data, L, dt)

Compute mean yaw update loss over an entire trajectory sequence.

# Arguments
- `trajectory_data::Matrix{Float64}`: N×3 matrix where each row is [yaw, v, delta]
- `L::Float64`: Wheelbase length of the vehicle (meters)
- `dt::Float64`: Time step duration (seconds)

# Returns
- `Float64`: Mean physics loss averaged over all trajectory steps

# Example
```julia
# Create a 10-step trajectory
trajectory = [yaw_angles velocities steering_angles]
loss = trajectory_yaw_loss(trajectory, 2.5, 0.1)
```
"""
function trajectory_yaw_loss(trajectory_data, L, dt)
    total_loss = 0.0
    n_steps = size(trajectory_data, 1) - 1
    
    for t in 1:n_steps
        # Extract current yaw, velocity, and steering angle
        yaw_t, v_t, delta_t = trajectory_data[t, :]
        yaw_t_next = trajectory_data[t+1, 1]
        
        # Accumulate loss for this timestep
        loss_t = yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
        total_loss += loss_t
    end
    
    return total_loss / n_steps
end


# ============================================================================
# TEST CASES — only executed when this file is run directly
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

println("="^70)
println("Testing Physics-Informed Yaw Update Loss Function")
println("="^70)
println()

# Vehicle parameters (typical small autonomous vehicle)
L = 2.5  # Wheelbase length in meters
dt = 0.1  # Time step in seconds

# Test 1: Straight line motion (zero steering angle, physics-compliant)
println("Test 1: Straight Line Motion (Zero Steering)")
println("-"^70)
yaw_t = 0.0                    # Current heading: 0 radians (pointing along x-axis)
v_t = 10.0                     # Velocity: 10 m/s
delta_t = 0.0                  # Steering angle: 0 (straight)
yaw_t_next = 0.0               # Yaw should remain 0

loss = yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
@printf("Initial yaw: %.4f rad (%.1f°)\n", yaw_t, rad2deg(yaw_t))
@printf("Velocity: %.2f m/s, Steering: %.4f rad (%.1f°), L: %.2f m, dt: %.2f s\n", 
        v_t, delta_t, rad2deg(delta_t), L, dt)
@printf("Next yaw: %.4f rad (%.1f°)\n", yaw_t_next, rad2deg(yaw_t_next))
@printf("Physics Loss: %.8f (should be ~0.0)\n", loss)
println()

# Test 2: Gentle left turn (positive steering, physics-compliant)
println("Test 2: Gentle Left Turn")
println("-"^70)
yaw_t = 0.0                    # Starting heading: 0 radians
v_t = 10.0                     # Velocity: 10 m/s
delta_t = 0.1                  # Small positive steering angle (left turn)
# yaw_rate = (10 / 2.5) * tan(0.1) ≈ 4.0 * 0.1003 ≈ 0.4012
# yaw_t_next = 0 + 0.4012 * 0.1 ≈ 0.04012
yaw_t_next = 0.04012

loss = yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
@printf("Initial yaw: %.4f rad (%.1f°)\n", yaw_t, rad2deg(yaw_t))
@printf("Velocity: %.2f m/s, Steering: %.4f rad (%.1f°), L: %.2f m, dt: %.2f s\n", 
        v_t, delta_t, rad2deg(delta_t), L, dt)
@printf("Next yaw: %.4f rad (%.1f°)\n", yaw_t_next, rad2deg(yaw_t_next))
@printf("Physics Loss: %.8f (should be ~0.0)\n", loss)
println()

# Test 3: Right turn (negative steering, physics-compliant)
println("Test 3: Right Turn")
println("-"^70)
yaw_t = π/2                    # Starting heading: π/2 (pointing along y-axis)
v_t = 5.0                      # Velocity: 5 m/s
delta_t = -0.15                # Negative steering angle (right turn)
# yaw_rate = (5 / 2.5) * tan(-0.15) ≈ 2.0 * (-0.1511) ≈ -0.3022
# yaw_t_next = π/2 + (-0.3022) * 0.1 ≈ 1.5708 - 0.03022 ≈ 1.54058
yaw_t_next = 1.54058

loss = yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
@printf("Initial yaw: %.4f rad (%.1f°)\n", yaw_t, rad2deg(yaw_t))
@printf("Velocity: %.2f m/s, Steering: %.4f rad (%.1f°), L: %.2f m, dt: %.2f s\n", 
        v_t, delta_t, rad2deg(delta_t), L, dt)
@printf("Next yaw: %.4f rad (%.1f°)\n", yaw_t_next, rad2deg(yaw_t_next))
@printf("Physics Loss: %.8f (should be ~0.0)\n", loss)
println()

# Test 4: Stationary vehicle (zero velocity, physics-compliant)
println("Test 4: Stationary Vehicle (v = 0)")
println("-"^70)
yaw_t = 0.0                    # Current heading: 0 radians
v_t = 0.0                      # Stationary (no velocity)
delta_t = 0.2                  # Steering angle doesn't matter when v=0
yaw_t_next = 0.0               # Yaw won't change if vehicle isn't moving

loss = yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
@printf("Initial yaw: %.4f rad (%.1f°)\n", yaw_t, rad2deg(yaw_t))
@printf("Velocity: %.2f m/s, Steering: %.4f rad (%.1f°), L: %.2f m, dt: %.2f s\n", 
        v_t, delta_t, rad2deg(delta_t), L, dt)
@printf("Next yaw: %.4f rad (%.1f°)\n", yaw_t_next, rad2deg(yaw_t_next))
@printf("Physics Loss: %.8f (should be ~0.0)\n", loss)
println()

# Test 5: Sharp turn at higher speed (physics-compliant)
println("Test 5: Sharp Turn at Higher Speed")
println("-"^70)
yaw_t = 0.0                    # Starting heading: 0 radians
v_t = 15.0                     # Higher velocity: 15 m/s
delta_t = 0.2                  # Sharper steering angle
# yaw_rate = (15 / 2.5) * tan(0.2) ≈ 6.0 * 0.2027 ≈ 1.2162
# yaw_t_next = 0 + 1.2162 * 0.1 ≈ 0.12162
yaw_t_next = 0.12162

loss = yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
@printf("Initial yaw: %.4f rad (%.1f°)\n", yaw_t, rad2deg(yaw_t))
@printf("Velocity: %.2f m/s, Steering: %.4f rad (%.1f°), L: %.2f m, dt: %.2f s\n", 
        v_t, delta_t, rad2deg(delta_t), L, dt)
@printf("Next yaw: %.4f rad (%.1f°)\n", yaw_t_next, rad2deg(yaw_t_next))
@printf("Physics Loss: %.8f (should be ~0.0)\n", loss)
println()

# Test 6: Physics violation (incorrect yaw change)
println("Test 6: Physics Violation (Incorrect Yaw)")
println("-"^70)
yaw_t = 0.0                    # Starting heading: 0 radians
v_t = 10.0                     # Velocity: 10 m/s
delta_t = 0.1                  # Steering angle: 0.1 rad
yaw_t_next = 0.2               # WRONG! Should be ~0.04012, not 0.2

loss = yaw_update_loss(yaw_t, yaw_t_next, v_t, delta_t, L, dt)
@printf("Initial yaw: %.4f rad (%.1f°)\n", yaw_t, rad2deg(yaw_t))
@printf("Velocity: %.2f m/s, Steering: %.4f rad (%.1f°), L: %.2f m, dt: %.2f s\n", 
        v_t, delta_t, rad2deg(delta_t), L, dt)
@printf("Next yaw: %.4f rad (%.1f°) [INCORRECT - should be ~0.04012 rad]\n", 
        yaw_t_next, rad2deg(yaw_t_next))
@printf("Physics Loss: %.8f (should be HIGH due to violation)\n", loss)
println()

# Test 7: Multi-step trajectory (straight → left turn → straight)
println("Test 7: Multi-Step Trajectory (Straight → Turn → Straight)")
println("-"^70)
println("Testing a 7-step maneuver...")

dt_traj = 0.1
L_traj = 2.5

# Create trajectory: straight, then turn left, then straight again
trajectory = zeros(8, 3)  # 8 timesteps, 3 variables [yaw, v, delta]

# Phase 1: Straight (steps 0-2)
trajectory[1, :] = [0.0, 10.0, 0.0]      # yaw=0, v=10, delta=0 (straight)
trajectory[2, :] = [0.0, 10.0, 0.0]      # No change
trajectory[3, :] = [0.0, 10.0, 0.0]      # No change

# Phase 2: Left turn (steps 3-5)
# delta=0.1, yaw_rate = (10/2.5)*tan(0.1) ≈ 0.4012
trajectory[4, :] = [0.0, 10.0, 0.1]      # Start turning
yaw_increment = (10.0 / 2.5) * tan(0.1) * 0.1  # ≈ 0.04012
trajectory[5, :] = [0.04012, 10.0, 0.1]
trajectory[6, :] = [0.08024, 10.0, 0.1]

# Phase 3: Straighten out (steps 6-7)
trajectory[7, :] = [0.12036, 10.0, 0.0]  # Back to straight
trajectory[8, :] = [0.12036, 10.0, 0.0]  # Continue straight

println("\nTrajectory data:")
println("Step | yaw (rad) | yaw (°)  | v (m/s) | delta (rad) | delta (°) | Phase")
println("-----|-----------|----------|---------|-------------|-----------|-------------")
for i in 1:8
    phase = i <= 3 ? "Straight" : (i <= 6 ? "Turning Left" : "Straight")
    @printf("%4d | %.5f   | %6.2f   | %.2f    | %.5f     | %6.2f    | %s\n", 
            i-1, trajectory[i,1], rad2deg(trajectory[i,1]), 
            trajectory[i,2], trajectory[i,3], rad2deg(trajectory[i,3]), phase)
end

traj_loss = trajectory_yaw_loss(trajectory, L_traj, dt_traj)
@printf("\nMean Trajectory Loss: %.8f (should be ~0.0)\n", traj_loss)
println()

println("="^70)
println("All tests complete!")
println("="^70)
println()
println("Note: The yaw update equation depends on:")
println("  • Velocity (v): Higher speed → faster yaw rate for same steering")
println("  • Steering angle (delta): Larger angle → sharper turn")
println("  • Wheelbase (L): Longer wheelbase → gentler turns for same steering")
println("  • Time step (dt): Larger dt → larger yaw change per step")

end # if abspath(PROGRAM_FILE) == @__FILE__
