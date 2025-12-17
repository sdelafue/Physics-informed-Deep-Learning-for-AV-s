using Statistics

# === GRID CONFIGURATION ===
const GRID_SIZE = 100
const MAX_RANGE = 10.0 # meters
const RESOLUTION = (MAX_RANGE * 2) / GRID_SIZE 

# === HELPER FUNCTIONS ===
function lidar_to_occupancy_grid(points)
    grid = zeros(Float32, GRID_SIZE, GRID_SIZE)
    for (x, y) in points
        if abs(x) > MAX_RANGE || abs(y) > MAX_RANGE; continue; end
        col = floor(Int, (x + MAX_RANGE) / RESOLUTION) + 1
        row = floor(Int, (y + MAX_RANGE) / RESOLUTION) + 1
        if 1 <= row <= GRID_SIZE && 1 <= col <= GRID_SIZE
            grid[row, col] = 1.0
        end
    end
    return grid
end

function read_nuscenes_lidar(file_path::String)
    """
    Reads a raw binary .pcd.bin file from nuScenes and returns a Vector of (x,y) tuples.
    Format: [x, y, z, intensity, ring_index] (5 Float32s per point)
    """
    raw_data = reinterpret(Float32, read(file_path))
    num_points = div(length(raw_data), 5)
    data_matrix = reshape(raw_data, 5, num_points)
    points_xy = [(data_matrix[1, i], data_matrix[2, i]) for i in 1:num_points]
    return points_xy
end

# === DATASET PATH AUTO-DETECTION ===
function find_dataset_path()
    """
    Attempts to automatically locate the nuScenes dataset in common locations.
    Returns the path if found, otherwise returns nothing.
    """
    possible_paths = [
        # Relative to current directory
        joinpath(pwd(), "samples", "LIDAR_TOP"),
        joinpath(pwd(), "v1.0-mini", "samples", "LIDAR_TOP"),
        joinpath(pwd(), "LIDAR_TOP"),
        
        # Common download locations
        joinpath(homedir(), "Downloads", "v1.0-mini", "samples", "LIDAR_TOP"),
        joinpath(homedir(), "Downloads", "samples", "LIDAR_TOP"),
        
        # Desktop
        joinpath(homedir(), "Desktop", "v1.0-mini", "samples", "LIDAR_TOP"),
    ]
    
    for path in possible_paths
        if isdir(path) && !isempty(filter(f -> endswith(f, ".bin"), readdir(path)))
            return path
        end
    end
    
    return nothing
end

function get_dataset_path()
    """
    Gets the dataset path through multiple fallback methods:
    1. Command-line argument
    2. Auto-detection in common locations
    3. User input prompt
    """
    # Method 1: Command-line argument
    if length(ARGS) > 0
        path = ARGS[1]
        # Clean up the path
        path = strip(path, ['"', '\''])
        path = replace(path, '\\' => '/')
        
        println("DEBUG: Received path: '$path'")
        println("DEBUG: isdir() result: $(isdir(path))")
        
        if isdir(path)
            println("✓ Using path from command-line argument: $path")
            return path
        else
            println("⚠ Warning: Provided path does not exist: $path")
            println("DEBUG: Current working directory: $(pwd())")
        end
    end
    
    # Method 2: Auto-detection
    auto_path = find_dataset_path()
    if !isnothing(auto_path)
        println("✓ Auto-detected dataset at: $auto_path")
        return auto_path
    end
    
    # Method 3: User input
    println("\n" * "="^60)
    println("📁 Dataset Path Required")
    println("="^60)
    println("Could not automatically locate the nuScenes dataset.")
    println("\nPlease enter the full path to the LIDAR_TOP folder.")
    println("\nExample paths:")
    println("  Windows: C:/Users/YourName/Downloads/v1.0-mini/samples/LIDAR_TOP")
    println("  Mac/Linux: /Users/YourName/Downloads/v1.0-mini/samples/LIDAR_TOP")
    println("\nTip: You can also run this script with the path as an argument:")
    println("  julia nuscenes_parser.jl <path_to_LIDAR_TOP>")
    println("="^60)
    print("\nEnter path: ")
    path = strip(readline())
    
    # Remove quotes if user added them
    path = strip(path, ['"', '\''])
    
    # Normalize path separators (Julia handles this, but be explicit)
    path = replace(path, '\\' => '/')
    
    # Validate the path
    if !isdir(path)
        error("❌ Invalid path: '$path'\nDirectory does not exist. Please check and try again.")
    end
    
    return path
end

# === MAIN EXECUTION ===
println("\n🚗 nuScenes LiDAR Parser")
println("="^60)

# Get dataset path using smart detection
dataset_path = get_dataset_path()

# Get list of all .bin files
bin_files = filter(f -> endswith(f, ".bin"), readdir(dataset_path, join=true))

println("\n✓ Found $(length(bin_files)) LiDAR scan(s)")

if !isempty(bin_files)
    # Process the first file
    test_file = bin_files[1]
    println("📊 Processing: $(basename(test_file))")
    
    # Parse binary data
    xy_points = read_nuscenes_lidar(test_file)
    println("  └─ Extracted $(length(xy_points)) points")
    
    # Convert to occupancy grid
    bev_grid = lidar_to_occupancy_grid(xy_points)
    println("  └─ Generated $(GRID_SIZE)x$(GRID_SIZE) BEV grid")
    
    # Visualize a center slice
    println("\n📈 BEV Grid Visualization (Center Slice):")
    println("="^60)
    center_col = 50
    for r in 40:2:60
        print("  ")
        for c in 45:55
            print(bev_grid[r, c] > 0 ? "█" : "·")
        end
        println()
    end
    println("="^60)
    println("\n✅ Success! Pipeline is working with real AV data.")
    println("💡 Tip: Modify the script to process all $(length(bin_files)) files.")
else
    error("❌ No .bin files found in the specified directory.\nPlease verify the path points to the LIDAR_TOP folder.")
end