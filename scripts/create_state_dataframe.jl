# Import necessary libraries
using CSV
using DataFrames
using JSON
using LinearAlgebra
using Statistics

# Define file paths
csv_filepath = joinpath("..", "datasets", "processed", "lidar_scans_sequential.csv")
ego_pose_filepath = joinpath("..", "datasets", "v1.0-mini", "ego_pose.json")

# Read the CSV file with sample tokens
println("Loading LiDAR samples CSV file from: ", csv_filepath)
df_lidar = CSV.read(csv_filepath, DataFrame)

println("Dataset loaded successfully!")
println("Total number of samples: ", nrow(df_lidar))

# Load ego_pose.json
println("\n" * "="^60)
println("Loading ego pose data...")
println("="^60)

ego_pose_data = JSON.parsefile(ego_pose_filepath)
println("Loaded ", length(ego_pose_data), " ego pose entries")

# Create a dictionary for fast lookup
ego_poses = Dict()
for pose in ego_pose_data
    ego_poses[pose["token"]] = pose
end

println("Created ego pose lookup dictionary")

# Function to convert quaternion to yaw angle
function quaternion_to_yaw(quat)
    """
    Convert quaternion [w, x, y, z] to yaw angle (rotation around z-axis)
    Returns yaw in radians
    """
    w, x, y, z = quat
    
    # Yaw (rotation around z-axis)
    yaw = atan(2.0 * (w*z + x*y), 1.0 - 2.0 * (y^2 + z^2))
    
    return yaw
end

# Extract ego vehicle states for all samples
println("\n" * "="^60)
println("Extracting ego vehicle states for all samples...")
println("="^60)

const WHEELBASE_LENGTH = 2.7

# Initialize arrays
ego_x = Float64[]
ego_y = Float64[]
ego_z = Float64[]
ego_yaw = Float64[]
ego_steering_angle = Float64
timestamps = Int64[]

prev_yaw = 0
# Extract position and yaw for each sample
for i in 1:nrow(df_lidar)
    ego_token = df_lidar[i, :ego_pose_token]
    ego_pose = ego_poses[ego_token]
    
    push!(ego_x, ego_pose["translation"][1])
    push!(ego_y, ego_pose["translation"][2])
    push!(ego_z, ego_pose["translation"][3])
    
    # Convert quaternion to yaw
    quat = ego_pose["rotation"]
    yaw = quaternion_to_yaw(quat)
    push!(ego_yaw, yaw)
    
    push!(timestamps, df_lidar[i, :timestamp])
end

# Add to DataFrame
df_lidar.ego_x = ego_x
df_lidar.ego_y = ego_y
df_lidar.ego_z = ego_z
df_lidar.ego_yaw = ego_yaw

println("Extracted positions and yaw for all ", nrow(df_lidar), " samples")

# Calculate velocities
println("\nCalculating velocities...")

vx = Float64[]
vy = Float64[]
speed = Float64[]

for i in 1:nrow(df_lidar)
    if i == 1
        push!(vx, NaN)
        push!(vy, NaN)
        push!(speed, NaN)
    else
        if df_lidar[i, :scene_token] == df_lidar[i-1, :scene_token]
            dt = (timestamps[i] - timestamps[i-1]) / 1e6
            dx = ego_x[i] - ego_x[i-1]
            dy = ego_y[i] - ego_y[i-1]
            
            vx_val = dx / dt
            vy_val = dy / dt
            speed_val = sqrt(vx_val^2 + vy_val^2)
            
            push!(vx, vx_val)
            push!(vy, vy_val)
            push!(speed, speed_val)
        else
            push!(vx, NaN)
            push!(vy, NaN)
            push!(speed, NaN)
        end
    end
end

df_lidar.vx = vx
df_lidar.vy = vy
df_lidar.speed = speed

println("Velocity calculation complete!")

# Calculate accelerations
println("\nCalculating accelerations...")

ax = Float64[]
ay = Float64[]
accel_magnitude = Float64[]

for i in 1:nrow(df_lidar)
    if i <= 1
        push!(ax, NaN)
        push!(ay, NaN)
        push!(accel_magnitude, NaN)
    elseif isnan(vx[i]) || isnan(vx[i-1])
        push!(ax, NaN)
        push!(ay, NaN)
        push!(accel_magnitude, NaN)
    else
        if df_lidar[i, :scene_token] == df_lidar[i-1, :scene_token]
            dt = (timestamps[i] - timestamps[i-1]) / 1e6
            dvx = vx[i] - vx[i-1]
            dvy = vy[i] - vy[i-1]
            
            ax_val = dvx / dt
            ay_val = dvy / dt
            accel_mag = sqrt(ax_val^2 + ay_val^2)
            
            push!(ax, ax_val)
            push!(ay, ay_val)
            push!(accel_magnitude, accel_mag)
        else
            push!(ax, NaN)
            push!(ay, NaN)
            push!(accel_magnitude, NaN)
        end
    end
end

df_lidar.ax = ax
df_lidar.ay = ay
df_lidar.accel = accel_magnitude

println("Acceleration calculation complete!")

# Calculate yaw rate (angular velocity around z-axis)
println("\nCalculating yaw rate...")

yaw_rate = Float64[]

for i in 1:nrow(df_lidar)
    if i == 1
        push!(yaw_rate, NaN)
    else
        if df_lidar[i, :scene_token] == df_lidar[i-1, :scene_token]
            dt = (timestamps[i] - timestamps[i-1]) / 1e6
            
            # Handle angle wrapping (e.g., from π to -π)
            dyaw = ego_yaw[i] - ego_yaw[i-1]
            
            # Normalize to [-π, π]
            while dyaw > π
                dyaw -= 2π
            end
            while dyaw < -π
                dyaw += 2π
            end
            
            yaw_rate_val = dyaw / dt
            push!(yaw_rate, yaw_rate_val)
        else
            push!(yaw_rate, NaN)
        end
    end
end

df_lidar.yaw_rate = yaw_rate

println("Yaw rate calculation complete!")

println("\nCalculating steering angle")

steering_angle = Float64[]

for i in 1:nrow(df_lidar)
    vel_magnitude = sqrt((df_lidar.vx[i] ^ 2) + (df_lidar.vy[i] ^ 2))
    str_angle = (WHEELBASE_LENGTH * df_lidar.yaw_rate[i]) / vel_magnitude
    push!(steering_angle, str_angle)
end

df_lidar.steering_angle = steering_angle

println("Steering angle calculation complete")

println("\nCalculating steering angle")

steering_rate = Float64[]

for i in 1:nrow(df_lidar)
    if i == 1
        push!(steering_rate, NaN)
    else
        dt = (timestamps[i] - timestamps[i-1]) / 1e6
        str_rate = (steering_angle[i] - steering_angle[i - 1]) / dt
        push!(steering_rate, str_rate)
    end
end

df_lidar.steering_rate = steering_rate

println("Steering angle rate calculation complete")

# Display results
println("\n" * "="^60)
println("First 10 samples with complete ego vehicle states:")
println("="^60)
println(first(df_lidar[:, [:sample_token, :scene_token, :ego_x, :ego_y, :ego_yaw, :speed, :accel, :yaw_rate]], 10))

# Display statistics
println("\n" * "="^60)
println("Ego Vehicle State Statistics:")
println("="^60)
println("Position:")
println("  X range: [", round(minimum(ego_x), digits=2), ", ", round(maximum(ego_x), digits=2), "] meters")
println("  Y range: [", round(minimum(ego_y), digits=2), ", ", round(maximum(ego_y), digits=2), "] meters")

valid_speeds = filter(!isnan, speed)
println("\nVelocity:")
println("  Mean speed: ", round(mean(valid_speeds), digits=2), " m/s (", round(mean(valid_speeds) * 3.6, digits=2), " km/h)")
println("  Max speed: ", round(maximum(valid_speeds), digits=2), " m/s")

valid_accels = filter(!isnan, accel_magnitude)
println("\nAcceleration:")
println("  Mean: ", round(mean(valid_accels), digits=2), " m/s²")
println("  Max: ", round(maximum(valid_accels), digits=2), " m/s²")

valid_yaw_rates = filter(!isnan, yaw_rate)
println("\nYaw Rate:")
println("  Mean: ", round(mean(abs.(valid_yaw_rates)), digits=4), " rad/s")
println("  Max: ", round(maximum(abs.(valid_yaw_rates)), digits=4), " rad/s")

valid_steering_angles = filter(!isnan, steering_angle)
println("\nSteering angle:")
println("  Mean: ", round(mean(abs.(valid_steering_angles)), digits=4), " rad/s")
println("  Max: ", round(maximum(abs.(valid_steering_angles)), digits=4), " rad/s")

valid_steering_rates = filter(!isnan, steering_rate)
println("\nSteering angle:")
println("  Mean: ", round(mean(abs.(valid_steering_rates)), digits=4), " rad/s")
println("  Max: ", round(maximum(abs.(valid_steering_rates)), digits=4), " rad/s")

println("\n" * "="^60)
println("Kinematic Bicycle Model State Variables Ready!")
println("="^60)
println("Available variables: x, y, yaw, velocity, acceleration, yaw_rate")

# Add this to the end of your existing script

# Save the processed DataFrame
println("\n" * "="^60)
println("Saving processed data...")
println("="^60)

output_filepath = joinpath("..", "datasets", "processed", "ego_vehicle_states.csv")
CSV.write(output_filepath, df_lidar)

println("Data saved to: ", output_filepath)
println("Total samples saved: ", nrow(df_lidar))

# Count valid samples (with velocity and acceleration)
valid_samples = sum(.!isnan.(df_lidar.speed) .& .!isnan.(df_lidar.accel))
println("Valid samples with complete state: ", valid_samples)

# Summary of what's available
println("\n" * "="^60)
println("Dataset Summary for Physics-Informed Model:")
println("="^60)
println("Available State Variables:")
println("  - Position: ego_x, ego_y, ego_z")
println("  - Orientation: ego_yaw, steering_angle (radians)")
println("  - Velocity: vx, vy, speed (m/s)")
println("  - Acceleration: ax, ay, accel (m/s²)")
println("  - Angular velocity: yaw_rate, steering_angle rate (rad/s)")
println("  - Wheelbase (L) - vehicle-specific constant")
println("  - Temporal: timestamp")