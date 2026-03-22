using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using Flux
using Statistics
using Random
using CSV
using DataFrames


# =================================================================
# 1. LOAD SAMPLE LIDAR SCANS FOR TESTING SCRIPT (TO BE MODIFIED LATER)
# =================================================================

LIDAR_SCAN_FILE_PATH = "datasets/LIDAR_TOP"








# =================================================================
# INSERT HELPER FUNCTIONS BELOW HERE
# =================================================================

# IMPORTANT: This struct is only for getting
# measurements from the NuScenes dataset.
# Real time measurement will be a different struct.
struct NuScenesLidarPoint
    x_m::Float32
    y_m::Float32
    z_m::Float32
    intensity::Float32
    ring_index::Float32
end

struct RPLidarPoint
    angle::Float32
    distance::Float32
    quality::UInt8
    x::Float32
    y::Float32
end