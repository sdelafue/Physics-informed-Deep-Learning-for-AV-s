"""
Physics-Informed Loss Function: Position Update (Kinematic Bicycle Model)
Senior Design Project - Autonomous Vehicle Trajectory Prediction

This module implements the position update constraints from the Kinematic Bicycle Model
as a loss function for physics-informed deep learning.
"""

using Printf

"""
    position_update_loss(x_t, y_t, x_t_next, y_t_next, v_t, yaw_t, dt)

Compute the physics-informed loss for position updates based on the Kinematic Bicycle Model.

The physics equations enforced are:
    x_{t+1} = x_t + v_t * cos(yaw_t) * dt
    y_{t+1} = y_t + v_t * sin(yaw_t) * dt

# Arguments
- `x_t::Float64`: Current x-coordinate position (meters)
- `y_t::Float64`: Current y-coordinate position (meters)
- `x_t_next::Float64`: Next timestep x-coordinate position (meters)
- `y_t_next::Float64`: Next timestep y-coordinate position (meters)
- `v_t::Float64`: Current velocity (meters/second)
- `yaw_t::Float64`: Current yaw angle/heading (radians, measured from x-axis)
- `dt::Float64`: Time step duration (seconds)

# Returns
- `Float64`: Combined squared residual loss (sum of x and y residuals squared)

# Physics Interpretation
The loss measures how much the observed position change (x_t_next - x_t, y_t_next - y_t)
deviates from what the kinematic bicycle model predicts. A loss of 0 means the motion
perfectly obeys the physics equations.

# Example
```julia
loss = position_update_loss(0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.1)
```
"""
function position_update_loss(x_t, y_t, x_t_next, y_t_next, v_t, yaw_t, dt)
    # Predict next positions based on kinematic bicycle model physics
    x_pred = x_t + v_t * cos(yaw_t) * dt
    y_pred = y_t + v_t * sin(yaw_t) * dt
    
    # Calculate residuals (difference between observed and physics-predicted)
    residual_x = x_t_next - x_pred
    residual_y = y_t_next - y_pred
    
    # Return squared residuals as loss (L2 norm)
    # You can modify this to use absolute values (L1 norm) if preferred
    return residual_x^2 + residual_y^2
end


"""
    trajectory_position_loss(trajectory_data, dt)

Compute mean position update loss over an entire trajectory sequence.

# Arguments
- `trajectory_data::Matrix{Float64}`: N×4 matrix where each row is [x, y, v, yaw]
- `dt::Float64`: Time step duration (seconds)

# Returns
- `Float64`: Mean physics loss averaged over all trajectory steps

# Example
```julia
# Create a 10-step trajectory
trajectory = [x_positions y_positions velocities yaw_angles]
loss = trajectory_position_loss(trajectory, 0.1)
```
"""
function trajectory_position_loss(trajectory_data, dt)
    total_loss = 0.0
    n_steps = size(trajectory_data, 1) - 1
    
    for t in 1:n_steps
        # Extract current and next state
        x_t, y_t, v_t, yaw_t = trajectory_data[t, :]
        x_t_next, y_t_next = trajectory_data[t+1, 1:2]
        
        # Accumulate loss for this timestep
        loss_t = position_update_loss(x_t, y_t, x_t_next, y_t_next, v_t, yaw_t, dt)
        total_loss += loss_t
    end
    
    return total_loss / n_steps
end


# ============================================================================
# TEST CASES — only executed when this file is run directly
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

println("="^70)
println("Testing Physics-Informed Position Update Loss Function")
println("="^70)
println()

# Test 1: Straight line motion along x-axis (physics-compliant)
println("Test 1: Straight Line Motion (Physics-Compliant)")
println("-"^70)
x_t, y_t = 0.0, 0.0           # Start at origin
v_t = 1.0                      # Velocity: 1 m/s
yaw_t = 0.0                    # Heading: along positive x-axis (0 radians)
dt = 0.1                       # Time step: 0.1 seconds
x_t_next = 0.1                 # After 0.1s at 1 m/s, should move 0.1m in x
y_t_next = 0.0                 # No y movement when heading along x-axis

loss = position_update_loss(x_t, y_t, x_t_next, y_t_next, v_t, yaw_t, dt)
@printf("Initial position: (%.2f, %.2f)\n", x_t, y_t)
@printf("Velocity: %.2f m/s, Yaw: %.2f rad, dt: %.2f s\n", v_t, yaw_t, dt)
@printf("Next position: (%.2f, %.2f)\n", x_t_next, y_t_next)
@printf("Physics Loss: %.6f (should be ~0.0)\n", loss)
println()

# Test 2: Straight line motion along y-axis (physics-compliant)
println("Test 2: Straight Line Motion Along Y-Axis (Physics-Compliant)")
println("-"^70)
x_t, y_t = 0.0, 0.0
v_t = 2.0                      # Velocity: 2 m/s
yaw_t = π/2                    # Heading: along positive y-axis (90 degrees)
dt = 0.1
x_t_next = 0.0                 # No x movement when heading along y-axis
y_t_next = 0.2                 # After 0.1s at 2 m/s, should move 0.2m in y

loss = position_update_loss(x_t, y_t, x_t_next, y_t_next, v_t, yaw_t, dt)
@printf("Initial position: (%.2f, %.2f)\n", x_t, y_t)
@printf("Velocity: %.2f m/s, Yaw: %.2f rad (90°), dt: %.2f s\n", v_t, yaw_t, dt)
@printf("Next position: (%.2f, %.2f)\n", x_t_next, y_t_next)
@printf("Physics Loss: %.6f (should be ~0.0)\n", loss)
println()

# Test 3: Diagonal motion at 45 degrees (physics-compliant)
println("Test 3: Diagonal Motion at 45° (Physics-Compliant)")
println("-"^70)
x_t, y_t = 0.0, 0.0
v_t = 1.414                    # Velocity: √2 m/s
yaw_t = π/4                    # Heading: 45 degrees
dt = 0.1
# At 45°: cos(π/4) = sin(π/4) = √2/2 ≈ 0.707
# So displacement = 1.414 * 0.707 * 0.1 ≈ 0.1 in both x and y
x_t_next = 0.1
y_t_next = 0.1

loss = position_update_loss(x_t, y_t, x_t_next, y_t_next, v_t, yaw_t, dt)
@printf("Initial position: (%.2f, %.2f)\n", x_t, y_t)
@printf("Velocity: %.2f m/s, Yaw: %.2f rad (45°), dt: %.2f s\n", v_t, yaw_t, dt)
@printf("Next position: (%.2f, %.2f)\n", x_t_next, y_t_next)
@printf("Physics Loss: %.6f (should be ~0.0)\n", loss)
println()

# Test 4: Physics violation - position doesn't match velocity
println("Test 4: Physics Violation (Incorrect Position)")
println("-"^70)
x_t, y_t = 0.0, 0.0
v_t = 1.0
yaw_t = 0.0
dt = 0.1
x_t_next = 0.5                 # Moved too far! (should be 0.1)
y_t_next = 0.0

loss = position_update_loss(x_t, y_t, x_t_next, y_t_next, v_t, yaw_t, dt)
@printf("Initial position: (%.2f, %.2f)\n", x_t, y_t)
@printf("Velocity: %.2f m/s, Yaw: %.2f rad, dt: %.2f s\n", v_t, yaw_t, dt)
@printf("Next position: (%.2f, %.2f) [INCORRECT - should be (0.1, 0.0)]\n", x_t_next, y_t_next)
@printf("Physics Loss: %.6f (should be HIGH due to violation)\n", loss)
println()

# Test 5: Trajectory loss over multiple steps
println("Test 5: Multi-Step Trajectory Loss")
println("-"^70)
println("Testing a 5-step straight-line trajectory...")

# Create a physics-compliant trajectory: vehicle moving at 1 m/s along x-axis
dt = 0.1
trajectory = zeros(6, 4)  # 6 timesteps, 4 variables [x, y, v, yaw]

for i in 1:6
    trajectory[i, 1] = (i-1) * 0.1  # x increases by 0.1 each step
    trajectory[i, 2] = 0.0          # y stays at 0
    trajectory[i, 3] = 1.0          # constant velocity
    trajectory[i, 4] = 0.0          # constant yaw (straight line)
end

println("Trajectory data:")
println("Step | x     | y     | v    | yaw")
println("-----|-------|-------|------|------")
for i in 1:6
    @printf("%4d | %.2f  | %.2f  | %.2f | %.2f\n", i-1, trajectory[i,1], trajectory[i,2], trajectory[i,3], trajectory[i,4])
end

traj_loss = trajectory_position_loss(trajectory, dt)
@printf("\nMean Trajectory Loss: %.6f (should be ~0.0)\n", traj_loss)
println()

println("="^70)
println("All tests complete!")
println("="^70)

end # if abspath(PROGRAM_FILE) == @__FILE__