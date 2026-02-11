using Pkg
parent_dir = dirname(@__DIR__)
print(parent_dir)
Pkg.activate(parent_dir)


using GLMakie
using Statistics
# === CONFIGURATION ===
const COLOR_BY = :height  # Options: :height, :intensity, :distance

# === HELPER FUNCTIONS ===
function read_nuscenes_lidar(file_path::String)
    """
    Reads a nuScenes .pcd.bin file and returns all 5 components.
    Returns: (x, y, z, intensity, ring_index) as separate vectors
    """
    raw_data = reinterpret(Float32, read(file_path))
    num_points = div(length(raw_data), 5)
    data_matrix = reshape(raw_data, 5, num_points)
    
    return (
        x = data_matrix[1, :],
        y = data_matrix[2, :],
        z = data_matrix[3, :],
        intensity = data_matrix[4, :],
        ring_index = data_matrix[5, :]
    )
end

function find_dataset_path()
    """Auto-detect dataset location"""
    possible_paths = [
        joinpath(pwd(), "datasets", "LIDAR_TOP"),
        joinpath(pwd(), "samples", "LIDAR_TOP"),
        joinpath(pwd(), "v1.0-mini", "samples", "LIDAR_TOP"),
        joinpath(homedir(), "Downloads", "v1.0-mini", "samples", "LIDAR_TOP"),
    ]
    
    for path in possible_paths
        if isdir(path) && !isempty(filter(f -> endswith(f, ".bin"), readdir(path)))
            return path
        end
    end
    return nothing
end

function get_dataset_path()
    """Get dataset path with fallback methods"""
    if length(ARGS) > 0
        path = strip(ARGS[1], ['"', '\''])
        if isdir(path)
            return path
        end
    end
    
    auto_path = find_dataset_path()
    if !isnothing(auto_path)
        println("✓ Found dataset at: $auto_path")
        return auto_path
    end
    
    println("\n📁 Please enter the path to your LIDAR_TOP folder:")
    path = strip(readline(), ['"', '\''])
    if !isdir(path)
        error("Invalid path: $path")
    end
    return path
end

# === MAIN VISUALIZATION ===
println("\n🎨 nuScenes Point Cloud Visualizer")
println("="^60)

# Get dataset
dataset_path = get_dataset_path()
bin_files = filter(f -> endswith(f, ".bin"), readdir(dataset_path, join=true))

if isempty(bin_files)
    error("No .bin files found!")
end

# Load a random sample
sample_file = rand(bin_files)
println("📊 Loading: $(basename(sample_file))")

data = read_nuscenes_lidar(sample_file)
println("  └─ Points: $(length(data.x))")
println("  └─ X range: [$(round(minimum(data.x), digits=1)), $(round(maximum(data.x), digits=1))] m")
println("  └─ Y range: [$(round(minimum(data.y), digits=1)), $(round(maximum(data.y), digits=1))] m")
println("  └─ Z range: [$(round(minimum(data.z), digits=1)), $(round(maximum(data.z), digits=1))] m")

# Compute colors based on selected mode
if COLOR_BY == :height
    colors = data.z
    color_label = "Height (m)"
elseif COLOR_BY == :intensity
    colors = data.intensity
    color_label = "Intensity"
else  # :distance
    colors = sqrt.(data.x.^2 .+ data.y.^2 .+ data.z.^2)
    color_label = "Distance (m)"
end

# Create 3D visualization
println("\n🖼️  Generating 3D visualization...")
fig = Figure(size=(1200, 800))

# 3D scatter plot
ax3d = Axis3(fig[1, 1], 
    title="nuScenes LiDAR Point Cloud",
    xlabel="X (forward, m)",
    ylabel="Y (left, m)",
    zlabel="Z (up, m)",
    aspect=:data
)

scatter!(ax3d, data.x, data.y, data.z,
    color=colors,
    colormap=:viridis,
    markersize=1.5,
    alpha=0.6
)

Colorbar(fig[1, 2], 
    limits=(minimum(colors), maximum(colors)),
    colormap=:viridis,
    label=color_label
)

# Bird's Eye View (XY plane)
ax_bev = Axis(fig[2, 1],
    title="Bird's Eye View (Top-Down)",
    xlabel="X (forward, m)",
    ylabel="Y (left, m)",
    aspect=DataAspect()
)

scatter!(ax_bev, data.x, data.y,
    color=colors,
    colormap=:viridis,
    markersize=2,
    alpha=0.5
)

# Statistics panel
stats_text = """
Dataset Statistics:
━━━━━━━━━━━━━━━━━━━
Total Points: $(length(data.x))
X: [$(round(minimum(data.x), digits=1)), $(round(maximum(data.x), digits=1))] m
Y: [$(round(minimum(data.y), digits=1)), $(round(maximum(data.y), digits=1))] m
Z: [$(round(minimum(data.z), digits=1)), $(round(maximum(data.z), digits=1))] m

Intensity: [$(round(minimum(data.intensity), digits=1)), $(round(maximum(data.intensity), digits=1))]
Ring Index: [$(Int(minimum(data.ring_index))), $(Int(maximum(data.ring_index)))]

Ground (~z < -1.5m): $(sum(data.z .< -1.5)) points
Above vehicle (~z > 0): $(sum(data.z .> 0)) points
"""

Label(fig[2, 2], stats_text, 
    tellwidth=true,        
    tellheight=false,
    width=300,             
    halign=:left,
    valign=:top,
    fontsize=12
)

println("✅ Visualization ready!")
println("\n💡 Controls:")
println("  - Left click + drag: Rotate 3D view")
println("  - Right click + drag: Pan")
println("  - Scroll: Zoom")
println("  - Close window to exit")

display(fig)

# Keep window open
println("\n⏳ Displaying... (close the window to exit)")
wait(display(fig))