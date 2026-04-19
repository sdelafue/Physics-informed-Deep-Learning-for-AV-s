
# Include order matters: data loader → model → report helpers → testing functions
include("utils/data_loading/data_loading_helper_functions.jl")
include("models/physics_informed_lstm.jl")
include("utils/generate_plots/report_metrics.jl")
include("utils/testing_functions/model_testing.jl")

# Run the example
println("=" ^ 60)
println("Testing Physics-Informed LSTM Model")
println("=" ^ 60)

model = train_on_real_data("Physics-informed LSTM")

println("\n" * "=" ^ 60)
println("Testing complete! Model trained successfully.")
println("=" ^ 60)