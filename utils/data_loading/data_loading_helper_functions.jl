using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using CSV
using DataFrames
using Random
using Printf

# ============================================================================
# PREPROCESSING PIPELINE FOR PHYSICS-INFORMED LSTM TRAINING DATA
# ============================================================================
#
# Pipeline flow:
#   load_and_filter_vehicles()  →  filtered DataFrame (vehicles only)
#   build_sequences()           →  Dict mapping instance_token → feature matrix
#   generate_input_mat()        →  input matrix for one sliding window position
#   generate_target_mat()       →  target matrix for one sliding window position
#   generate_sample_list()      →  shuffled Vector of (input, target) tuples
#
# Feature columns (7 KBM variables + dt = 8 columns):
#   Column 1: dt        [seconds]  — time elapsed since previous timestep
#   Column 2: x         [meters]   — global x position
#   Column 3: y         [meters]   — global y position
#   Column 4: yaw       [radians]  — heading angle
#   Column 5: vx        [m/s]      — velocity in global x direction
#   Column 6: vy        [m/s]      — velocity in global y direction
#   Column 7: yaw_rate  [rad/s]    — angular velocity
#
# Matrix dimensions:
#   Input matrix:  (n_input,  7) — e.g., (4, 7) for 4 observed timesteps
#   Target matrix: (n_target, 7) — e.g., (6, 7) for 6 predicted timesteps
#
# Sample list structure:
#   Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}
#   Each tuple: (input_matrix, target_matrix)
#   List is shuffled randomly for training
# ============================================================================

# Vehicle categories that obey the Kinematic Bicycle Model
const VEHICLE_CATEGORIES = [
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus.rigid",
    "vehicle.bus.bendy",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "vehicle.construction",
    "vehicle.trailer"
]

# Feature columns to extract from the object DataFrame
# NOTE: obj_x, obj_y, obj_yaw are the column names in object_state_table.csv
#       ego_x, ego_y, ego_yaw are the column names in ego_vehicle_states.csv
const OBJ_FEATURE_COLS = [:obj_x, :obj_y, :obj_yaw, :vx, :vy, :yaw_rate]
const EGO_FEATURE_COLS = [:ego_x, :ego_y, :ego_yaw, :vx, :vy, :yaw_rate]

"""
    load_and_filter_vehicles(obj_csv_path::String) → DataFrame

Loads the object_state_table.csv and filters to vehicle categories only.
Drops rows where velocity data is missing (first timestep per instance).
Returns a cleaned DataFrame sorted by instance and time.
"""
function load_and_filter_vehicles(obj_csv_path::String)
    df = CSV.read(obj_csv_path, DataFrame)

    # Filter to vehicle categories only
    vehicle_mask = [cat in VEHICLE_CATEGORIES for cat in df.category]
    vehicles = df[vehicle_mask, :]

    # Sort by instance then timestamp for correct temporal ordering
    sort!(vehicles, [:instance_token, :timestamp])

    # Drop rows with NaN velocity (first timestep per instance has no prior
    # state to compute derivatives from)
    vehicles = dropmissing(vehicles, [:vx, :vy, :yaw_rate])
    # Also handle NaN stored as actual NaN in CSV (not Missing)
    valid = .!isnan.(vehicles.vx) .& .!isnan.(vehicles.vy) .& .!isnan.(vehicles.yaw_rate)
    vehicles = vehicles[valid, :]

    return vehicles
end

"""
    load_and_prepare_ego(ego_csv_path::String) → DataFrame

Loads the ego_vehicle_states.csv and prepares it for sequence building.
Drops rows where velocity data is missing (first timestep per scene).
The ego vehicle gets a synthetic instance_token per scene so it can be
processed identically to other vehicles in build_sequences().
"""
function load_and_prepare_ego(ego_csv_path::String)
    df = CSV.read(ego_csv_path, DataFrame)

    sort!(df, [:scene_token, :timestamp])

    # Create a synthetic instance_token for each scene's ego vehicle
    # so the ego can be processed through the same pipeline
    df.instance_token = "ego_" .* df.scene_token

    # Rename ego columns to match object column names for uniform processing
    rename!(df, :ego_x => :obj_x, :ego_y => :obj_y, :ego_yaw => :obj_yaw)

    # Drop rows with NaN velocity
    valid_rows = .!isnan.(df.vx) .& .!isnan.(df.vy) .& .!isnan.(df.yaw_rate)
    df = df[valid_rows, :]

    return df
end

"""
    compute_dt(timestamps::Vector) → Vector{Float64}

Computes the time delta (in seconds, 6 decimal places precision) between
consecutive timestamps. The nuScenes timestamps are in microseconds.
Returns a vector of length n where the first element is 0.0 (no prior step).
"""
function compute_dt(timestamps::AbstractVector)
    n = length(timestamps)
    dt = zeros(Float64, n)
    for i in 2:n
        # Convert microsecond difference to seconds, round to 6 decimal places
        dt[i] = round((timestamps[i] - timestamps[i-1]) / 1e6; digits=6)
    end
    return dt
end

"""
    build_sequences(df::DataFrame; min_length::Int=10) → Dict{String, Matrix{Float64}}

Groups the DataFrame by instance_token and builds a feature matrix for each
instance. Each matrix has shape (timesteps, 7) with columns:
    [dt, x, y, yaw, vx, vy, yaw_rate]

Only instances with at least `min_length` timesteps are included.
The first row of each sequence is dropped because dt=0.0 provides no
temporal information to the model.

Returns: Dict mapping instance_token → feature matrix
"""
function build_sequences(df::DataFrame; min_length::Int=10)
    sequences = Dict{String, Matrix{Float64}}()

    for group in groupby(df, :instance_token)
        # Sort by time (should already be sorted, but enforce it)
        sorted_group = sort(group, :timestamp)

        # Compute dt for this instance's timeline
        dt = compute_dt(sorted_group.timestamp)

        # Build feature matrix: [dt, x, y, yaw, vx, vy, yaw_rate]
        n = nrow(sorted_group)
        mat = Matrix{Float64}(undef, n, 7)
        mat[:, 1] = dt
        mat[:, 2] = sorted_group.obj_x
        mat[:, 3] = sorted_group.obj_y
        mat[:, 4] = sorted_group.obj_yaw
        mat[:, 5] = sorted_group.vx
        mat[:, 6] = sorted_group.vy
        mat[:, 7] = sorted_group.yaw_rate

        # Drop the first row (dt=0.0, no temporal context)
        mat = mat[2:end, :]

        # Only keep sequences long enough for the sliding window
        instance_id = first(group.instance_token)
        if size(mat, 1) >= min_length
            sequences[instance_id] = mat
        end
    end

    return sequences
end

"""
    generate_input_mat(sequence::Matrix{Float64}, start_idx::Int, n_input::Int) → Matrix{Float64}

Extracts the input matrix from a sequence at a given starting index.

Returns: Matrix of shape (n_input, 7)
    Rows:    timesteps [1, 2, ..., n_input]
    Columns: [dt, x, y, yaw, vx, vy, yaw_rate]
    Units:   [s,  m, m, rad, m/s, m/s, rad/s]
"""
function generate_input_mat(sequence::Matrix{Float64}, start_idx::Int, n_input::Int)
    return sequence[start_idx:start_idx + n_input - 1, :]
end

"""
    generate_target_mat(sequence::Matrix{Float64}, start_idx::Int, n_input::Int, n_target::Int) → Matrix{Float64}

Extracts the target (ground truth) matrix from a sequence, starting
immediately after the input window.

Returns: Matrix of shape (n_target, 7)
    Rows:    timesteps [1, 2, ..., n_target]
    Columns: [dt, x, y, yaw, vx, vy, yaw_rate]
    Units:   [s,  m, m, rad, m/s, m/s, rad/s]
"""
function generate_target_mat(sequence::Matrix{Float64}, start_idx::Int, n_input::Int, n_target::Int)
    target_start = start_idx + n_input
    return sequence[target_start:target_start + n_target - 1, :]
end

"""
    generate_sample_list(sequences::Dict{String, Matrix{Float64}};
                         n_input::Int=4, n_target::Int=6,
                         stride::Int=1, seed::Int=42) → Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}

Applies a sliding window across all sequences to produce a shuffled list
of (input, target) training samples.

Sliding window parameters:
    - window_size = n_input + n_target (total timesteps consumed per sample)
    - stride = step size between consecutive windows (1 = maximum overlap)

Returns: Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}
    Length:  total number of samples across all instances
    Each tuple contains:
        tuple[1] = input_matrix  — shape (n_input,  7) = (4, 7) by default
        tuple[2] = target_matrix — shape (n_target, 7) = (6, 7) by default
    
    Matrix column layout (identical for input and target):
        Col 1: dt        [seconds]   — time delta from previous timestep
        Col 2: x         [meters]    — global x position
        Col 3: y         [meters]    — global y position
        Col 4: yaw       [radians]   — heading angle
        Col 5: vx        [m/s]       — velocity in global x
        Col 6: vy        [m/s]       — velocity in global y
        Col 7: yaw_rate  [rad/s]     — angular velocity

    The list is randomly shuffled so training batches are not biased
    toward any particular instance or scene.
"""
function generate_sample_list(sequences::Dict{String, Matrix{Float64}};
                              n_input::Int=4, n_target::Int=6,
                              stride::Int=1, seed::Int=42)
    window_size = n_input + n_target
    samples = Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}()

    for (instance_id, seq) in sequences
        n_timesteps = size(seq, 1)

        # Slide the window across this sequence
        start_idx = 1
        while start_idx + window_size - 1 <= n_timesteps
            input_mat  = generate_input_mat(seq, start_idx, n_input)
            target_mat = generate_target_mat(seq, start_idx, n_input, n_target)
            push!(samples, (input_mat, target_mat))
            start_idx += stride
        end
    end

    # Shuffle samples for unbiased training batches
    Random.seed!(seed)
    shuffle!(samples)

    return samples
end

# ============================================================================
# MAIN: Run the full pipeline
# ============================================================================

function main()
    # ---- Paths ----
    obj_csv = joinpath(@__DIR__, "..", "..", "datasets", "processed", "object_state_table.csv")
    ego_csv = joinpath(@__DIR__, "..", "..", "datasets", "processed", "ego_vehicle_states.csv")

    # ---- Hyperparameters ----
    N_INPUT  = 4   # observed timesteps (≈ 2.0 seconds at 0.5s intervals)
    N_TARGET = 6   # predicted timesteps (≈ 3.0 seconds)
    STRIDE   = 1   # sliding window stride
    MIN_SEQ  = N_INPUT + N_TARGET  # minimum sequence length to be usable

    # ---- Step 1: Load and filter data ----
    println("Loading object data...")
    obj_df = load_and_filter_vehicles(obj_csv)
    println("  → $(nrow(obj_df)) vehicle rows, $(length(unique(obj_df.instance_token))) instances")

    println("Loading ego data...")
    ego_df = load_and_prepare_ego(ego_csv)
    println("  → $(nrow(ego_df)) ego rows, $(length(unique(ego_df.instance_token))) scenes")

    # ---- Step 2: Combine ego + object vehicles into unified DataFrame ----
    # Select only the columns both DataFrames share for sequence building
    shared_cols = [:instance_token, :timestamp, :obj_x, :obj_y, :obj_yaw, :vx, :vy, :yaw_rate]
    combined_df = vcat(obj_df[:, shared_cols], ego_df[:, shared_cols])
    println("Combined: $(nrow(combined_df)) total rows")

    # ---- Step 3: Build per-instance sequences ----
    println("\nBuilding sequences (min_length=$MIN_SEQ)...")
    sequences = build_sequences(combined_df; min_length=MIN_SEQ)
    println("  → $(length(sequences)) usable sequences")

    # ---- Step 4: Generate shuffled training samples ----
    println("\nGenerating training samples (input=$N_INPUT, target=$N_TARGET, stride=$STRIDE)...")
    samples = generate_sample_list(sequences;
                                   n_input=N_INPUT, n_target=N_TARGET, stride=STRIDE)
    println("  → $(length(samples)) total samples")

    # ---- Summary ----
    println("\n" * "="^60)
    println("PIPELINE SUMMARY")
    println("="^60)
    println("Input matrix shape:  ($N_INPUT, 7)")
    println("Target matrix shape: ($N_TARGET, 7)")
    println("Features: [dt(s), x(m), y(m), yaw(rad), vx(m/s), vy(m/s), yaw_rate(rad/s)]")
    println("Total training samples: $(length(samples))")
    println()

    # ---- Sanity check: print first sample ----
    if !isempty(samples)
        input_ex, target_ex = samples[1]
        println("Example sample (first in shuffled list):")
        println("  Input matrix ($(size(input_ex))):")
        for row in 1:size(input_ex, 1)
            vals = join([@sprintf("%.6f", input_ex[row, col]) for col in 1:size(input_ex, 2)], "  ")
            println("    [$vals]")
        end
        println("  Target matrix ($(size(target_ex))):")
        for row in 1:size(target_ex, 1)
            vals = join([@sprintf("%.6f", target_ex[row, col]) for col in 1:size(target_ex, 2)], "  ")
            println("    [$vals]")
        end
    end

    return samples
end

# main() is called explicitly by train_on_real_data() in model_testing.jl
# It is not auto-run here so this file can be safely included without side effects