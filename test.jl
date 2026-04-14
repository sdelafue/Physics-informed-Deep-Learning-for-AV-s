
# Include order matters: data loader → model → testing functions
include("utils/data_loading/data_loading_helper_functions.jl")
include("models/lstm_baseline.jl")
include("utils/testing_functions/model_testing.jl")

# Run the example
println("=" ^ 60)
println("Testing LSTM Baseline Model")
println("=" ^ 60)

model = train_on_real_data()

println("\n" * "=" ^ 60)
println("Testing complete! Model trained successfully.")
println("=" ^ 60)