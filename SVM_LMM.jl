using Pkg
Pkg.add("TSAnalysis")

using TSAnalysis, Plots

# Load some time series data (replace with your dataset)
data = TSAnalysis.load_example("AirPassengers")

# Fit an ARIMA model (order: ARIMA(p,d,q))
model_arima = fit(ARIMA, data, (2, 1, 2)) # Example order (2,1,2)
arima_forecast = forecast(model_arima, 12)  # Forecast next 12 periods

# Plot the forecast
plot(data, label="Actual")
plot!(arima_forecast, label="ARIMA Forecast")


# Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q,m), m is seasonality period
model_sarima = fit(SARIMA, data, (1, 1, 1)(0, 1, 1, 12))  # SARIMA with monthly seasonality
sarima_forecast = forecast(model_sarima, 12)

plot(data, label="Actual")
plot!(sarima_forecast, label="SARIMA Forecast")


# Holt-Winters exponential smoothing
model_hw = fit(HoltWinters, data)
hw_forecast = forecast(model_hw, 12)

# Plotting the forecast
plot(data, label="Actual")
plot!(hw_forecast, label="Holt-Winters Forecast")


Pkg.add("MLJ")
Pkg.add("DecisionTree")
Pkg.add("TimeSeries")


using MLJ, TimeSeries, CSV, DataFrames

# Load your time series data (replace with your dataset)
data = CSV.read("your_time_series.csv", DataFrame)

# Create a time series problem
y = data[:Target]  # The target column (e.g., sales, stock price, etc.)
X = data[:, Not(:Target)]  # Other features

# Define the machine learning model (DecisionTreeRegressor as an example)
tree_model = @load DecisionTreeRegressor
model = tree_model()

# Train/test split
train_size = Int(0.8 * nrow(data))
X_train, X_test = X[1:train_size, :], X[train_size+1:end, :]
y_train, y_test = y[1:train_size], y[train_size+1:end]

# Fit the model
mach = machine(model, X_train, y_train)
fit!(mach)

# Make predictions
y_pred = predict(mach, X_test)

# Evaluate model performance
performance = rms(y_test, y_pred)
println("Root Mean Square Error: $performance")


# Load AutoML interface
Pkg.add("MLJFlux")  # For neural networks if needed

using MLJ, MLJFlux

# Create an AutoML pipeline (e.g., for neural networks)
automl_model = @load NeuralNetworkRegressor
automl_machine = machine(automl_model(), X_train, y_train)

fit!(automl_machine)
predictions = predict(automl_machine, X_test)

# Performance evaluation
println("RMSE for AutoML model: ", rms(predictions, y_test))


using LIBSVM  # SVM implementation in Julia
using CSV     # To read datasets in CSV format
using DataFrames  # Organize data in tabular form

# Load dataset from a CSV file
data = CSV.read("your_data.csv", DataFrame)

# Shuffle the data to randomize before splitting
shuffle!(data)

# Define the proportion for the training set (80% of the data)
train_ratio = 0.8
n = nrow(data)  # Get the total number of rows in the dataset
train_size = Int(floor(train_ratio * n))  # Number of training samples

# Split the dataset into training and testing sets
train_data = data[1:train_size, :]
test_data = data[train_size+1:end, :]

# Extract features (inputs) and labels (targets) from the training set
X_train = Matrix(train_data[:, ["Feature1", "Feature2"]])  # Feature matrix
y_train = Vector(train_data[:, "Label"])  # Target vector

# Extract features and labels from the test set
X_test = Matrix(test_data[:, ["Feature1", "Feature2"]])
y_test = Vector(test_data[:, "Label"])

# Train the SVM model using a linear kernel
model = LIBSVM.fit!(X_train, y_train; kernel=LIBSVM.Kernel.Linear)

# Print model details
println("Trained SVM Model: ", model)

# Make predictions on the test data
y_pred = predict(model, X_test)

# Display the predicted and actual labels
println("Predicted Labels: ", y_pred)
println("True Labels: ", y_test)

# Calculate the accuracy of the model
accuracy = sum(y_pred .== y_test) / length(y_test)
println("Accuracy: ", accuracy * 100, "%")

############################################################################################

# We need LinearAlgebra for basic vector operations
using LinearAlgebra

# Function to initialize weight vector (w) and bias (b)
function initialize_params(n_features)
    w = randn(n_features) * 0.01  # Small random weights
    b = 0.0  # Initialize bias as 0
    return w, b
end

# Prediction function (w'x + b)
function predict(w, b, X)
    return X * w .+ b  # Linear combination of inputs and weights
end

# Sign function for classification
function classify(pred)
    return pred .>= 0.0  # Classify as 1 if pred >= 0, else -1
end

# Hinge loss function (primal objective)
function hinge_loss(w, b, X, y, C)
    n = length(y)  # Number of samples
    # Compute the regularization term (L2 norm of w)
    reg_term = 0.5 * dot(w, w)
    
    # Compute the hinge loss
    margins = 1 .- y .* (X * w .+ b)
    hinge_term = sum(max(0, margin) for margin in margins)

    # Final loss (sum of hinge loss + regularization term)
    return reg_term + C * hinge_term / n
end

# Function to compute gradients and update w and b
function gradient_step!(w, b, X, y, C, lr)
    n = length(y)  # Number of samples
    margins = 1 .- y .* (X * w .+ b)
    
    # Gradient of the regularization term
    dw = w  # Gradient of 0.5 * ||w||^2

    # For data points where hinge loss is non-zero (i.e., misclassified points)
    for i in 1:n
        if margins[i] > 0
            dw -= C * y[i] * X[i, :]
            b -= C * y[i]
        end
    end

    # Update parameters (gradient descent step)
    w -= lr * dw / n
    b -= lr * b / n
end

# Main function to train the SVM
function train_svm(X, y; C=1.0, lr=0.001, epochs=1000)
    n_samples, n_features = size(X)
    
    # Initialize weights and bias
    w, b = initialize_params(n_features)
    
    # Training loop
    for epoch in 1:epochs
        # Update weights and bias using gradient descent
        gradient_step!(w, b, X, y, C, lr)

        # Optional: Print loss every 100 epochs
        if epoch % 100 == 0
            current_loss = hinge_loss(w, b, X, y, C)
            println("Epoch $epoch: Loss = $current_loss")
        end
    end
    
    return w, b  # Return the learned parameters
end

using Random

# Generate some synthetic data (linearly separable for simplicity)
function generate_data(n_samples)
    X = randn(n_samples, 2)  # Random 2D points
    y = [if x[1] + x[2] > 0 1 else -1 end for x in eachrow(X)]  # Simple decision boundary
    return X, y
end

# Load synthetic dataset
X, y = generate_data(100)

# Train SVM model
w, b = train_svm(X, y; C=1.0, lr=0.001, epochs=1000)

# Make predictions
y_pred = predict_svm(w, b, X)

# Calculate accuracy
println("Accuracy: ", accuracy(y, y_pred), "%")



############################

using LinearAlgebra
using Random
using Plots

# Generate synthetic data for mixed effects model
function generate_data(n_groups, n_per_group, σ_random, σ_noise)
    Random.seed!(123)
    n_samples = n_groups * n_per_group
    group_effects = randn(n_groups) * σ_random
    X = randn(n_samples)
    β = 2.0
    group = repeat(1:n_groups, inner=n_per_group)
    y = [β * X[i] + group_effects[group[i]] + randn() * σ_noise for i in 1:n_samples]
    return X, y, group
end

n_groups = 10
n_per_group = 20
σ_random = 1.0
σ_noise = 0.5
X, y, group = generate_data(n_groups, n_per_group, σ_random, σ_noise)

# Initialize model parameters
function initialize_params(n_groups)
    β = randn()
    b = randn(n_groups)
    σ_b = 1.0
    σ_noise = 0.5
    return β, b, σ_b, σ_noise
end

β, b, σ_b, σ_noise = initialize_params(n_groups)

# Log-likelihood function
function log_likelihood(X, y, group, β, b, σ_b, σ_noise)
    n = length(y)
    group_effects = [b[group[i]] for i in 1:n]
    residuals = y .- (X .* β .+ group_effects)
    ll_random = -0.5 * sum((b ./ σ_b).^2)
    ll_res

end 