using Pkg
parent_dir = dirname(@__DIR__)
Pkg.activate(parent_dir)

using GLMakie
using CairoMakie
using Statistics
using JLD2

# === CONFIGURATION ===
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

struct LidarXY
    lidar_scan::LidarScan
    x::Float32
    y::Float32
end

# === HELPER FUNCTIONS ===
function get_xy(scans::Vector{LidarScan})
    points = LidarXY[]

    for scan in scans
        for i in 1:scan.n
            r = scan.ranges[i]

            if r <= scan.range_min || r >= scan.range_max
                continue
            end

            θ = scan.angle_min + (i-1) * scan.angle_inc
            x = r * cos(θ)
            y = r * sin(θ)

            push!(points, LidarXY(scan, x, y))
        end
    end

    return points
end

function radius_filter_2d(points::Vector{LidarXY}, radius::Float32=0.1f0, min_neighbors::Int=2)
    n = length(points)
    if n == 0
        return points
    end

    filtered = LidarXY[]
    r2 = radius * radius

    for i in 1:n
        xi = points[i].x
        yi = points[i].y
        neighbor_count = 0

        for j in 1:n
            if i == j continue end

            dx = xi - points[j].x
            dy = yi - points[j].y

            if (dx*dx + dy*dy) <= r2
                neighbor_count += 1

                if neighbor_count >= min_neighbors
                    push!(filtered, points[i])
                    break
                end
            end
        end
    end

    return filtered
end

function sor_filter_2d(points::Vector{LidarXY}, k::Int=8, std_mul::Float32=1.0f0)
    n = length(points)
    if n <= k
        return points 
    end

    dist_to_neighbors = zeros(Float32, n)
    for i in 1:n
        dists = Float32[]
        xi = points[i].x
        yi = points[i].y
        for j in 1:n
            if i == j continue end
            xj = points[j].x
            yj = points[j].y
            push!(dists, sqrt((xi-xj)^2 + (yi-yj)^2))
        end
        sort!(dists)
        dist_to_neighbors[i] = mean(dists[1:k])
    end
    μ = mean(dist_to_neighbors)
    σ = std(dist_to_neighbors)

    # Keep points that are not outliers
    filtered = [p for (i,p) in enumerate(points) if dist_to_neighbors[i] <= μ + std_mul * σ]

    return filtered
end

function get_custom_lidar_scan_dir()
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
    else
        scans = first(values(data))
    end

    return collect(scans)
end

function lidar_scan_stats(scan::LidarScan, ranges)
    valid_count = length(ranges)
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
    Range Min: $(round(minimum(ranges), digits=2)) m
    Range Max: $(round(maximum(ranges), digits=2)) m
    Range Mean: $(round(mean(ranges), digits=2)) m
    Angle Min: $(round(rad2deg(scan.angle_min), digits=1)) deg
    Angle Inc: $(round(rad2deg(scan.angle_inc), digits=3)) deg
    Timestamp: $(round(scan.timestamp, digits=3))
    """
end

function create_lidar_scan_figure(points::Vector{LidarXY}; title::AbstractString="LiDAR Scan")
    fig = Figure(size=(1200, 800))

    ax_scan = Axis(
        fig[1, 1],
        title=title,
        xlabel="X (m)",
        ylabel="Y (m)",
        aspect=DataAspect()
    )

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    ranges = [sqrt(p.x^2 + p.y^2) for p in points]

    if !isempty(points)

        scatter!(
            ax_scan,
            xs,
            ys,
            color=ranges,
            colormap=:viridis,
            markersize=7
        )

        limits!(
            ax_scan,
            minimum(xs) - 0.5,
            maximum(xs) + 0.5,
            minimum(ys) - 0.5,
            maximum(ys) + 0.5
        )

        Colorbar(
            fig[1, 2],
            limits=(minimum(ranges), maximum(ranges)),
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

    scatter!(ax_profile, 1:length(ranges), ranges, markersize=5, color=:steelblue)

    Label(
        fig[2, 2],
        lidar_scan_stats(points[1].lidar_scan, ranges),
        tellwidth=true,
        tellheight=false,
        width=300,
        halign=:left,
        valign=:top,
        fontsize=12
    )

    return fig
end

function visualize_lidar_scan(scan::Vector{LidarXY}; title::AbstractString="LiDAR Scan")
    fig = create_lidar_scan_figure(scan; title=title)
    screen = display(fig)
    return (fig = fig, screen = screen)
end

function visualize_lidar_scans_from_file(file_path::AbstractString, scans::Vector{LidarXY}, scan_index::Int=1)
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

function visualize_all_lidar_scans(file_path::AbstractString, scans::Vector{Vector{LidarXY}})
    base_name = splitext(basename(file_path))[1]
    output_dir = joinpath(EXPORTED_DATA_DIR, base_name)
    mkpath(output_dir)

    println("Exporting $(length(scans)) scans from $(basename(file_path)) to $output_dir")

    for (index, scan_points) in enumerate(scans)
        fig = create_lidar_scan_figure(
            scan_points;
            title="$(base_name) | scan $index / $(length(scans))"
        )

        output_file = joinpath(output_dir, "scan_$(index).png")
        save(output_file, fig)
    end

    println("Finished exporting $(length(scans)) scans for $(basename(file_path))")
    return output_dir
end

# === MAIN PIPELINE ===
println("\nFiltered LiDAR Visualizer")
println("="^60)

scan_dir = get_custom_lidar_scan_dir()
jld2_files = sort(filter(f -> endswith(lowercase(f), ".jld2"), readdir(scan_dir, join=true)))

if isempty(jld2_files)
    error("No .jld2 files found")
end

println("Found $(length(jld2_files)) files")

sample_file = first(jld2_files)
unfiltered_scans = load_lidar_scans_from_jld2(sample_file)

println("Length of unfiltered: $(length(unfiltered_scans))")

println("Processing scans with statistical outlier removal filter...")

filtered_scans_xy = Vector{Vector{LidarXY}}()

for scan in unfiltered_scans
    xy_points = get_xy([scan])
    filtered_xy = sor_filter_2d(xy_points)

    push!(filtered_scans_xy, filtered_xy)
end

if isempty(filtered_scans_xy)
    error("No scans after filtering")
end

# === VISUALIZE SAMPLE ===
println("Visualizing filtered sample scan...")

sample_points = filtered_scans_xy[1]
sample = visualize_lidar_scan(sample_points)

println("Visualization ready")
println("Exporting all scans from every .jld2 file in custom_lidar_scans")

visualize_all_lidar_scans(jld2_files[1], filtered_scans_xy)

println("Finished exporting LiDAR scan PNGs for all .jld2 files")
println("Close the visualization window to exit")

wait(sample_view.screen)