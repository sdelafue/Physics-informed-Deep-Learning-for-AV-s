using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using Flux
using Statistics
using Random
using CSV
using DataFrames

# Load the main code
include("lstm_baseline.jl")

# Run the example
println("=" ^ 60)
println("Testing LSTM Baseline Model")
println("=" ^ 60)

model = example_usage()

println("\n" * "=" ^ 60)
println("Testing complete! Model trained successfully.")
println("=" ^ 60)