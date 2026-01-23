# Import necessary libraries
using CSV
using DataFrames
using JSON
using LinearAlgebra
using Statistics

# Define file paths
csv_filepath = joinpath("..","datasets", "processed", "lidar_scans_sequential.csv")
annotation_filepath = joinpath("..","datasets", "v1.0-mini", "sample_annotation.json")
instance_filepath = joinpath("..","datasets", "v1.0-mini", "instance.json")
category_filepath = joinpath("..","datasets", "v1.0-mini", "category.json")

# Read the CSV file with sample tokens
println("Loading LiDAR samples CSV file from: ", csv_filepath)
df_lidar = CSV.read(csv_filepath, DataFrame)

println("Dataset loaded successfully!")
println("Total number of samples: ", nrow(df_lidar))

# Load sample_annotation.json
println("\n" * "="^60)
println("Loading object annotation data...")
println("="^60)

annotation_data = JSON.parsefile(annotation_filepath)
println("Loaded ", length(annotation_data), " annotations")

# Load instance.json
instance_data = JSON.parsefile(instance_filepath)
instances = Dict(inst["token"] => inst for inst in instance_data)
println("Loaded ", length(instance_data), " instances")

# Load category.json (contains category names)
category_data = JSON.parsefile(category_filepath)
categories = Dict(cat["token"] => cat["name"] for cat in category_data)
println("Loaded ", length(category_data), " categories")

# Group annotations by sample_token
println("Grouping annotations by sample...")
sample_annotations = Dict{String, Vector{Any}}()
for ann in annotation_data
    sample_token = ann["sample_token"]
    if !haskey(sample_annotations, sample_token)
        sample_annotations[sample_token] = []
    end
    push!(sample_annotations[sample_token], ann)
end

println("Annotations grouped by sample")

# Function to convert quaternion to yaw angle
function quaternion_to_yaw(quat)
    w, x, y, z = quat
    yaw = atan(2.0 * (w*z + x*y), 1.0 - 2.0 * (y^2 + z^2))
    return yaw
end

# Extract object states for all samples
println("\n" * "="^60)
println("Extracting object states...")
println("="^60)

# Initialize arrays for object states
obj_sample_tokens = String[]
obj_timestamps = Int64[]
obj_scene_tokens = String[]
obj_instance_tokens = String[]
obj_categories = String[]
obj_x = Float64[]
obj_y = Float64[]
obj_z = Float64[]
obj_yaw = Float64[]

# Extract position and yaw for each object in each sample
for i in 1:nrow(df_lidar)
    
    sample_token = df_lidar[i, :sample_token]
    timestamp = df_lidar[i, :timestamp]
    scene_token = df_lidar[i, :scene_token]
    
    # Get annotations for this sample
    if haskey(sample_annotations, sample_token)
        objects = sample_annotations[sample_token]
        
        for obj in objects
            # Get instance info
            instance_token = obj["instance_token"]
            instance = instances[instance_token]
            
            # Get category name from category_token
            category_token = instance["category_token"]
            category = categories[category_token]
            
            # Extract position and rotation
            position = obj["translation"]
            rotation = obj["rotation"]
            yaw = quaternion_to_yaw(rotation)
            
            # Store data
            push!(obj_sample_tokens, sample_token)
            push!(obj_timestamps, timestamp)
            push!(obj_scene_tokens, scene_token)
            push!(obj_instance_tokens, instance_token)
            push!(obj_categories, category)
            push!(obj_x, position[1])
            push!(obj_y, position[2])
            push!(obj_z, position[3])
            push!(obj_yaw, yaw)
            
        end
    end
end

println("Extracted states for ", length(obj_sample_tokens), " object instances across all samples")

# Create DataFrame with object positions
df_objects = DataFrame(
    sample_token = obj_sample_tokens,
    timestamp = obj_timestamps,
    scene_token = obj_scene_tokens,
    instance_token = obj_instance_tokens,
    category = obj_categories,
    obj_x = obj_x,
    obj_y = obj_y,
    obj_z = obj_z,
    obj_yaw = obj_yaw
)

# Sort by instance_token and timestamp to group same objects together
sort!(df_objects, [:instance_token, :timestamp])

println("\nDataFrame created with ", nrow(df_objects), " object states")
println("Unique objects tracked: ", length(unique(df_objects.instance_token)))

# Calculate velocities for each object
println("\n" * "="^60)
println("Calculating object velocities...")
println("="^60)

vx = Float64[]
vy = Float64[]
speed = Float64[]

for i in 1:nrow(df_objects)
    if i == 1
        # First row overall
        push!(vx, NaN)
        push!(vy, NaN)
        push!(speed, NaN)
    else
        # Check if same object and same scene
        if (df_objects[i, :instance_token] == df_objects[i-1, :instance_token] &&
            df_objects[i, :scene_token] == df_objects[i-1, :scene_token])
            
            # Calculate time difference (microseconds to seconds)
            dt = (df_objects[i, :timestamp] - df_objects[i-1, :timestamp]) / 1e6
            
            # Calculate position differences
            dx = df_objects[i, :obj_x] - df_objects[i-1, :obj_x]
            dy = df_objects[i, :obj_y] - df_objects[i-1, :obj_y]
            
            # Calculate velocity
            vx_val = dx / dt
            vy_val = dy / dt
            speed_val = sqrt(vx_val^2 + vy_val^2)
            
            push!(vx, vx_val)
            push!(vy, vy_val)
            push!(speed, speed_val)
        else
            # Different object or scene boundary
            push!(vx, NaN)
            push!(vy, NaN)
            push!(speed, NaN)
        end
    end
end

# Add velocities to DataFrame
df_objects.vx = vx
df_objects.vy = vy
df_objects.speed = speed

println("Velocity calculation complete!")

# Calculate accelerations for each object
println("\n" * "="^60)
println("Calculating object accelerations...")
println("="^60)

ax = Float64[]
ay = Float64[]
accel_magnitude = Float64[]

for i in 1:nrow(df_objects)
    if i <= 1
        push!(ax, NaN)
        push!(ay, NaN)
        push!(accel_magnitude, NaN)
    elseif isnan(vx[i]) || isnan(vx[i-1])
        push!(ax, NaN)
        push!(ay, NaN)
        push!(accel_magnitude, NaN)
    else
        # Check if same object and same scene
        if (df_objects[i, :instance_token] == df_objects[i-1, :instance_token] &&
            df_objects[i, :scene_token] == df_objects[i-1, :scene_token])
            
            # Calculate time difference
            dt = (df_objects[i, :timestamp] - df_objects[i-1, :timestamp]) / 1e6
            
            # Calculate velocity differences
            dvx = vx[i] - vx[i-1]
            dvy = vy[i] - vy[i-1]
            
            # Calculate acceleration
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

# Add accelerations to DataFrame
df_objects.ax = ax
df_objects.ay = ay
df_objects.accel = accel_magnitude

println("Acceleration calculation complete!")

# Calculate yaw rate for each object
println("\n" * "="^60)
println("Calculating object yaw rates...")
println("="^60)

yaw_rate = Float64[]

for i in 1:nrow(df_objects)
    if i == 1
        push!(yaw_rate, NaN)
    else
        # Check if same object and same scene
        if (df_objects[i, :instance_token] == df_objects[i-1, :instance_token] &&
            df_objects[i, :scene_token] == df_objects[i-1, :scene_token])
            
            # Calculate time difference
            dt = (df_objects[i, :timestamp] - df_objects[i-1, :timestamp]) / 1e6
            
            # Calculate yaw difference with angle wrapping
            dyaw = df_objects[i, :obj_yaw] - df_objects[i-1, :obj_yaw]
            
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

# Add yaw rate to DataFrame
df_objects.yaw_rate = yaw_rate

println("Yaw rate calculation complete!")

# Display summary statistics
println("\n" * "="^60)
println("Object State Extraction Summary:")
println("="^60)
println("Total object observations: ", nrow(df_objects))
println("Unique objects: ", length(unique(df_objects.instance_token)))
println("Unique categories: ", unique(df_objects.category))

# Count by category
println("\nObjects by category:")
for cat in sort(unique(df_objects.category))
    count = sum(df_objects.category .== cat)
    unique_objs = length(unique(df_objects[df_objects.category .== cat, :instance_token]))
    println("  ", cat, ": ", count, " observations (", unique_objs, " unique objects)")
end

# Display first 10 rows
println("\n" * "="^60)
println("First 10 object states:")
println("="^60)
println(first(df_objects[:, [:instance_token, :category, :obj_x, :obj_y, :obj_yaw, :speed, :accel, :yaw_rate]], 10))

# Statistics for valid data
valid_speeds = filter(!isnan, df_objects.speed)
valid_accels = filter(!isnan, df_objects.accel)
valid_yaw_rates = filter(!isnan, df_objects.yaw_rate)

println("\n" * "="^60)
println("Object Motion Statistics:")
println("="^60)
if length(valid_speeds) > 0
    println("Speed:")
    println("  Mean: ", round(mean(valid_speeds), digits=2), " m/s")
    println("  Max: ", round(maximum(valid_speeds), digits=2), " m/s")
end

if length(valid_accels) > 0
    println("Acceleration:")
    println("  Mean: ", round(mean(valid_accels), digits=2), " m/s²")
    println("  Max: ", round(maximum(valid_accels), digits=2), " m/s²")
end

if length(valid_yaw_rates) > 0
    println("Yaw Rate:")
    println("  Mean: ", round(mean(abs.(valid_yaw_rates)), digits=4), " rad/s")
    println("  Max: ", round(maximum(abs.(valid_yaw_rates)), digits=4), " rad/s")
end

# Save to CSV
println("\n" * "="^60)
println("Saving object states to CSV...")
println("="^60)

output_filepath = joinpath("..","datasets", "processed", "object_state_table.csv")
CSV.write(output_filepath, df_objects)

println("Object states saved to: ", output_filepath)
println("Total rows: ", nrow(df_objects))
println("\n" * "="^60)
println("Object state extraction complete!")
println("="^60)
