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

# Load instance.json (contains object category info)
instance_data = JSON.parsefile(instance_filepath)
instances = Dict(inst["token"] => inst for inst in instance_data)

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
total_objects = 0