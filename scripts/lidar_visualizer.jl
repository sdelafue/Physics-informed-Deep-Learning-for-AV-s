using Pkg
parent_dir = dirname(@__DIR__)
print(parent_dir)
Pkg.activate(parent_dir)

using GLMakie
using CairoMakie
using Statistics
using JLD2

# === CONFIGURATION ===
const COLOR_BY = :height  # Options: :height, :intensity, :distance
const CUSTOM_SCAN_DIR = joinpath(parent_dir, "datasets", "custom_lidar_scans")
const EXPORTED_DATA_DIR = joinpath(parent_dir, "exported_data")

struct LidarScan
    timestamp::Float64
    n::Int
    angle_min::Float32
    angle_inc::Float32
    range_min::Float32
    range_max::Float32
    ranges::Vector{Float32}
end

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
        println("Found dataset at: $auto_path")
        return auto_path
    end

    println("\nPlease enter the path to your LIDAR_TOP folder:")
    path = strip(readline(), ['"', '\''])
    if !isdir(path)
        error("Invalid path: $path")
    end
    return path
end

function get_custom_lidar_scan_dir()
    if length(ARGS) > 0
        path = strip(ARGS[1], ['"', '\''])
        if isdir(path)
            return path
        end
    end

    if isdir(CUSTOM_SCAN_DIR)
        println("Found custom LiDAR scan directory at: $CUSTOM_SCAN_DIR")
        return CUSTOM_SCAN_DIR
    end

    error("Custom LiDAR scan directory not found: $CUSTOM_SCAN_DIR")
end

function load_lidar_scans_from_jld2(file_path::AbstractString)
    data = load(file_path)

    if haskey(data, "scans")
        scans = data["scans"]
    elseif !isempty(keys(data))
        first_key = first(collect(keys(data)))
        scans = data[first_key]
    else
        error("No datasets found in $file_path")
    end

    if !(scans isa AbstractVector)
        error("Expected a vector of LiDAR scans in $file_path")
    end

    return collect(scans)
end

function get_valid_scan_points(scan::LidarScan)
    count = min(scan.n, length(scan.ranges))
    if count == 0
        empty_vector = Float32[]
        return (
            x = empty_vector,
            y = empty_vector,
            ranges = empty_vector,
            angles = empty_vector
        )
    end

    angles = scan.angle_min .+ (0:(count - 1)) .* scan.angle_inc
    ranges = scan.ranges[1:count]
    valid_mask = isfinite.(ranges) .& (ranges .>= scan.range_min) .& (ranges .<= scan.range_max)

    valid_ranges = ranges[valid_mask]
    valid_angles = angles[valid_mask]

    x = valid_ranges .* cos.(valid_angles)
    y = valid_ranges .* sin.(valid_angles)

    return (
        x = x,
        y = y,
        ranges = valid_ranges,
        angles = valid_angles
    )
end

function lidar_scan_stats(scan::LidarScan, points)
    valid_count = length(points.ranges)
    total_count = min(scan.n, length(scan.ranges))

    if valid_count == 0
        return """
        Scan Statistics:
        Total Rays: $total_count
        Valid Returns: 0
        Timestamp: $(round(scan.timestamp, digits=3))
        """
    end

    return """
    Scan Statistics:
    Total Rays: $total_count
    Valid Returns: $valid_count
    Range Min: $(round(minimum(points.ranges), digits=2)) m
    Range Max: $(round(maximum(points.ranges), digits=2)) m
    Range Mean: $(round(mean(points.ranges), digits=2)) m
    Angle Min: $(round(rad2deg(scan.angle_min), digits=1)) deg
    Angle Inc: $(round(rad2deg(scan.angle_inc), digits=3)) deg
    Timestamp: $(round(scan.timestamp, digits=3))
    """
end

function create_lidar_scan_figure(scan::LidarScan; title::AbstractString="LiDAR Scan")
    points = get_valid_scan_points(scan)

    fig = Figure(size=(1200, 800))

    ax_scan = Axis(
        fig[1, 1],
        title=title,
        xlabel="X (m)",
        ylabel="Y (m)",
        aspect=DataAspect()
    )

    if !isempty(points.x)
        scatter!(
            ax_scan,
            points.x,
            points.y,
            color=points.ranges,
            colormap=:viridis,
            markersize=7
        )

        limits!(
            ax_scan,
            minimum(points.x) - 0.5,
            maximum(points.x) + 0.5,
            minimum(points.y) - 0.5,
            maximum(points.y) + 0.5
        )

        Colorbar(
            fig[1, 2],
            limits=(minimum(points.ranges), maximum(points.ranges)),
            colormap=:viridis,
            label="Range (m)"
        )
    else
        text!(ax_scan, 0.0, 0.0, text="No valid returns", align=(:center, :center))
    end

    ax_profile = Axis(
        fig[2, 1],
        title="Range Profile",
        xlabel="Beam Index",
        ylabel="Range (m)"
    )

    if !isempty(points.ranges)
        scatter!(ax_profile, 1:length(points.ranges), points.ranges, markersize=5, color=:steelblue)
    end

    Label(
        fig[2, 2],
        lidar_scan_stats(scan, points),
        tellwidth=true,
        tellheight=false,
        width=300,
        halign=:left,
        valign=:top,
        fontsize=12
    )

    return fig
end

function visualize_lidar_scan(scan::LidarScan; title::AbstractString="LiDAR Scan")
    fig = create_lidar_scan_figure(scan; title=title)
    screen = display(fig)
    return (fig = fig, screen = screen)
end

function visualize_lidar_scans_from_file(file_path::AbstractString; scan_index::Int=1)
    scans = load_lidar_scans_from_jld2(file_path)

    if isempty(scans)
        error("No LiDAR scans found in $file_path")
    end

    bounded_index = clamp(scan_index, 1, length(scans))
    scan = scans[bounded_index]
    title = "$(basename(file_path)) | scan $bounded_index / $(length(scans))"

    println("Visualizing $title")
    println("  Valid points: $(length(get_valid_scan_points(scan).x))")

    return visualize_lidar_scan(scan; title=title)
end

function visualize_all_lidar_scans(file_path::AbstractString)
    scans = load_lidar_scans_from_jld2(file_path)
    base_name = splitext(basename(file_path))[1]
    output_dir = joinpath(EXPORTED_DATA_DIR, base_name)
    mkpath(output_dir)

    println("Exporting $(length(scans)) scans from $(basename(file_path)) to $output_dir")

    for (index, scan) in enumerate(scans)
        fig = create_lidar_scan_figure(
            scan;
            title="$(base_name) | scan $index / $(length(scans))"
        )
        output_file = joinpath(output_dir, "scan_$(index).png")
        save(output_file, fig)
    end

    println("Finished exporting $(length(scans)) scans for $(basename(file_path))")
    return output_dir
end

# === MAIN VISUALIZATION ===
println("\nCustom LiDAR Scan Visualizer")
println("="^60)

scan_dir = get_custom_lidar_scan_dir()
jld2_files = sort(filter(f -> endswith(lowercase(f), ".jld2"), readdir(scan_dir, join=true)))

if isempty(jld2_files)
    error("No .jld2 files found in $scan_dir")
end

println("Found $(length(jld2_files)) .jld2 LiDAR files")

sample_file = first(jld2_files)
sample_scans = load_lidar_scans_from_jld2(sample_file)

if isempty(sample_scans)
    error("No LiDAR scans found in $(basename(sample_file))")
end

println("Loading sample visualization from $(basename(sample_file))")
println("  Total scans in file: $(length(sample_scans))")

sample_scan = first(sample_scans)
sample_points = get_valid_scan_points(sample_scan)

println("  Sample scan valid returns: $(length(sample_points.x))")
println("  Range bounds: [$(round(sample_scan.range_min, digits=2)), $(round(sample_scan.range_max, digits=2))] m")
println("  Angle bounds: [$(round(rad2deg(sample_scan.angle_min), digits=2)), $(round(rad2deg(sample_scan.angle_min + (sample_scan.n - 1) * sample_scan.angle_inc), digits=2))] deg")

sample_view = visualize_lidar_scans_from_file(sample_file; scan_index=1)

println("Visualization ready")
println("Exporting all scans from every .jld2 file in custom_lidar_scans")

for file_path in jld2_files
    visualize_all_lidar_scans(file_path)
end

println("Finished exporting LiDAR scan PNGs for all .jld2 files")
println("Close the visualization window to exit")

wait(sample_view.screen)
