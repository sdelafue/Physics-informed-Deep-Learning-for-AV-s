using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using Flux
using Statistics
using Random
using CSV
using DataFrames

# File paths for the model and testing functions
model_filepath = "models/lstm_baseline.jl"
test_filepath = "utils/testing_functions/model_testing.jl"
include(model_filepath)
include(test_filepath)

# Run the example
println("=" ^ 60)
println("Testing LSTM Baseline Model")
println("=" ^ 60)

model = example_usage()

println("\n" * "=" ^ 60)
println("Testing complete! Model trained successfully.")
println("=" ^ 60)