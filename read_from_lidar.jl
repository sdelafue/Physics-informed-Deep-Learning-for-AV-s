using Sockets
using JLD2

const HOST = ip"127.0.0.1"
const PORT = 16000
const MAX_SCANS = 100
const OUTPUT_FILE = "lidar_scans.jld2"

struct LidarScan
    timestamp::Float64
    n::Int
    angle_min::Float32
    angle_inc::Float32
    range_min::Float32
    range_max::Float32
    ranges::Vector{Float32}
end

# Read exactly N bytes from a TCP stream (TCP may return partial reads)
function readexact(sock::TCPSocket, n::Int)
    buf = Vector{UInt8}(undef, n)
    i = 1
    while i <= n
        got = readbytes!(sock, view(buf, i:n), n - i + 1)
        got == 0 && error("Socket closed")
        i += got
    end
    return buf
end

function save_scans(scans::Vector{LidarScan}, filename::AbstractString)
    @save filename scans
    println("Saved $(length(scans)) LiDAR scans to $filename")
end

sock = connect(HOST, PORT)
println("Connected to tcp://$HOST:$PORT")

scans = LidarScan[]

try
    while length(scans) < MAX_SCANS
        # 1) Read frame length (uint32 little-endian)
        len_bytes = readexact(sock, 4)
        nbytes = reinterpret(UInt32, len_bytes)[1]

        # 2) Read payload
        payload = readexact(sock, Int(nbytes))

        # 3) Decode payload header
        n = Int(reinterpret(UInt32, payload[1:4])[1])
        angle_min = reinterpret(Float32, payload[5:8])[1]
        angle_inc = reinterpret(Float32, payload[9:12])[1]
        range_min = reinterpret(Float32, payload[13:16])[1]
        range_max = reinterpret(Float32, payload[17:20])[1]

        # 4) Decode ranges
        ranges_view = reinterpret(Float32, payload[21:end])
        if length(ranges_view) != n
            println("Mismatch: header n=$n ranges=$(length(ranges_view))")
            continue
        end

        # Copy ranges into a Vector{Float32} so each scan owns its data cleanly
        ranges = collect(ranges_view)

        scan = LidarScan(
            time(),
            n,
            angle_min,
            angle_inc,
            range_min,
            range_max,
            ranges,
        )
        push!(scans, scan)

        # Print proof-of-life (a few values)
        mid = clamp(Int(floor(n / 2)), 1, n)
        println(
            "scan $(length(scans))/$MAX_SCANS | n=$n | ang_min=$angle_min | " *
            "ang_inc=$angle_inc | r1=$(ranges[1]) rmid=$(ranges[mid]) rlast=$(ranges[end])"
        )
    end

    save_scans(scans, OUTPUT_FILE)
finally
    close(sock)
    println("Socket closed")
end
