
#!/usr/bin/env julia

using Parquet
using DataFrames
using GLMakie

# --------- Helpers to find and load parquet files ---------

"Return a list of .parquet files in the given directory and print them."
function list_parquet_files(dir::AbstractString)
    files = filter(f -> endswith(f, ".parquet"), readdir(dir; join=true))
    isempty(files) && error("No .parquet files found in directory: $dir")

    println("Found $(length(files)) parquet files:")
    for (i, f) in enumerate(files)
        println("[$i] ", basename(f))
    end
    return files
end

"Load a parquet file into a DataFrame; optionally truncate for speed."
function load_parquet(path::AbstractString; max_rows::Int = 1_000_000)
    println("Loading parquet file: $path")
    pf = Parquet.File(path)
    df = DataFrame(pf)

    println("File has $(nrow(df)) rows and $(ncol(df)) columns.")
    println("Column names: ", names(df))

    if nrow(df) > max_rows
        println("Too many rows for comfortable visualization; keeping first $max_rows rows.")
        df = df[1:max_rows, :]
    end

    return df
end

"Try to guess which columns are x, y, z, intensity, and frame."
function guess_columns(df::DataFrame)
    cols = names(df)

    # Candidate names (add more if needed)
    candidates = Dict(
        :x          => [:x, :X, :px, :pos_x],
        :y          => [:y, :Y, :py, :pos_y],
        :z          => [:z, :Z, :pz, :pos_z, :height],
        :intensity  => [:intensity, :reflectance, :int],
        :frame      => [:frame, :frame_id, :frame_idx, :timestamp, :time]
    )

    chosen = Dict{Symbol,Union{Symbol,Nothing}}(
        :x => nothing, :y => nothing, :z => nothing,
        :intensity => nothing, :frame => nothing,
    )

    for (target, names_try) in candidates
        for name_try in names_try
            if name_try in cols
                chosen[target] = name_try
                break
            end
        end
    end

    println("\nGuessed columns (change in code if wrong):")
    for (k, v) in chosen
        println("  $(k): ", v)
    end

    return chosen
end

# --------- Visualization ---------

"Visualize LiDAR points in 3D, optionally with per-frame slider."
function visualize_points(df::DataFrame;
                          xcol::Symbol,
                          ycol::Symbol,
                          zcol::Symbol,
                          intensity_col::Union{Symbol,Nothing}=nothing,
                          frame_col::Union{Symbol,Nothing}=nothing)

    if !(xcol in names(df) && ycol in names(df) && zcol in names(df))
        error("x, y, z columns not found in DataFrame. Check column names.")
    end

    if frame_col !== nothing && !(frame_col in names(df))
        @warn "frame_col $frame_col not found, falling back to single-cloud view."
        frame_col = nothing
    end

    if intensity_col !== nothing && !(intensity_col in names(df))
        @warn "intensity_col $intensity_col not found, ignoring intensity."
        intensity_col = nothing
    end

    GLMakie.activate!()

    if frame_col === nothing
        # ---- Single point cloud ----
        xs = df[!, xcol]
        ys = df[!, ycol]
        zs = df[!, zcol]

        points = Point3f.(xs, ys, zs)

        fig = Figure(resolution=(900, 800))
        ax = Axis3(fig[1, 1], title="LiDAR point cloud", perspectiveness=0.8,
                   xlabel=string(xcol), ylabel=string(ycol), zlabel=string(zcol))

        if intensity_col === nothing
            meshscatter!(ax, points; markersize=1)
        else
            intensities = df[!, intensity_col]
            meshscatter!(ax, points; markersize=1, color=intensities, colormap=:viridis)
            Colorbar(fig[1, 2], label=string(intensity_col))
        end

        fig[2, 1] = Label(fig, "Drag with mouse to rotate, scroll to zoom.",
                          tellwidth=false)
        display(fig)

    else
        # ---- Multiple frames with slider ----
        frames = unique(df[!, frame_col])
        sort!(frames)  # ensure ordered
        nframes = length(frames)

        println("Detected $nframes unique frames based on column $(frame_col).")

        frame_obs = Observable(frames[1])

        fig = Figure(resolution=(900, 900))
        ax = Axis3(fig[1, 1], title="Frame $(frames[1])",
                   xlabel=string(xcol), ylabel=string(ycol), zlabel=string(zcol),
                   perspectiveness=0.8)

        # Observables that recompute points (and colors) when frame changes
        points_obs = lift(frame_obs) do f
            sub = df[df[!, frame_col] .== f, :]
            Point3f.(sub[!, xcol], sub[!, ycol], sub[!, zcol])
        end

        if intensity_col === nothing
            plt = meshscatter!(ax, points_obs; markersize=1)
        else
            color_obs = lift(frame_obs) do f
                sub = df[df[!, frame_col] .== f, :]
                sub[!, intensity_col]
            end
            plt = meshscatter!(ax, points_obs; markersize=1,
                               color=color_obs, colormap=:viridis)
            Colorbar(fig[1, 2], plt, label=string(intensity_col))
        end

        # Slider to move between frames
        slider = Slider(fig[2, 1], range=1:nframes, startvalue=1)
        on(slider.value) do i
            frame_obs[] = frames[i]
            ax.title = "Frame $(frames[i])"
        end

        fig[3, 1] = Label(fig,
            "Use slider to change frame. Drag to rotate, scroll to zoom.",
            tellwidth=false)

        display(fig)
    end
end

# --------- Main entry point ---------

function main()
    dir = length(ARGS) >= 1 ? ARGS[1] : "."
    files = list_parquet_files(dir)

    print("\nEnter index of file to open: ")
    idx = parse(Int, readline())
    @assert 1 <= idx <= length(files) "Invalid index"

    df = load_parquet(files[idx])

    # Try to guess columns (adjust manually if wrong)
    guessed = guess_columns(df)

    xcol = guessed[:x]      === nothing ? error("Could not guess x column") : guessed[:x]
    ycol = guessed[:y]      === nothing ? error("Could not guess y column") : guessed[:y]
    zcol = guessed[:z]      === nothing ? error("Could not guess z column") : guessed[:z]
    intensity_col = guessed[:intensity]
    frame_col     = guessed[:frame]

    visualize_points(df;
        xcol=xcol, ycol=ycol, zcol=zcol,
        intensity_col=intensity_col,
        frame_col=frame_col
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
