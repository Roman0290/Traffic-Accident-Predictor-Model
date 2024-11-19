import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



# Load the data
df = pd.read_excel("Cleaned.xlsx")  
df.head(3)

# Define mappings for categorical columns
weather_condition = {
    "Normal": 0,
    "Rainy": 1,
    "Cloudy": 2,
    "Windy": 3,
    "Snow": 4,
    "Fog or mist": 5,
    "Rainy and Windy": 6,
}

Cause_of_accident_mapping = {
    "Over Speeding": 0,
    "Reckless Driving": 1,
    "Priority denial": 2,
    "Law Breaking": 3,
    "Distance keeping": 4,
    "Moving Backward": 5,
    "Overtaking": 6,
    "Overloading": 7,
    "Changing lane to the left": 8,
    "Changing lane to the right": 9,
    "No distancing": 10,
    "Getting off the vehicle improperly": 11,
    "Improper parking": 12,
    "Driving carelessly": 13,
    "Driving at high speed": 14,
    "Driving to the left": 15,
    "Overturning": 16,
    "Turnover": 17,
    "Driving under the influence of drugs": 18,
    "Drunk driving": 19,
}

Light_conditions_mapping = {
    "Daylight": 0,
    "Darkness - with light": 1,
    "Darkness - no lighting": 2
}

Accident_severity_mapping = {
    "Minimal injury crash": 0,
    "Minor injury crash": 1,
    "Major injury crash": 2
}

# Apply mappings to the DataFrame
df['Weather_conditions'] = df['Weather_conditions'].map(weather_condition)
df['Cause_of_accident'] = df['Cause_of_accident'].map(Cause_of_accident_mapping)
df['Light_conditions'] = df['Light_conditions'].map(Light_conditions_mapping)
df['Accident_severity'] = df['Accident_severity'].map(Accident_severity_mapping)

df.head(3)


# Drop any rows with NaN values that could arise from unmapped categories
df.dropna(inplace=True)

# Separate features (X) and target (y)
X = df.drop(columns=['Accident_severity'])
Y = df['Accident_severity']

# Display the features and target arrays
print("Features (X):", X[:5])  # Show first 5 rows for illustration
print("Target (y):", Y[:5])

# Train-validation-test split
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = []

    def _add_intercept(self, X):
        """Add a column of ones to X for the intercept term."""
        return np.c_[np.ones(X.shape[0]), X]

    def _hypothesis(self, X):
        """Compute the linear hypothesis (predictions) based on X and current theta."""
        return np.dot(X, self.theta)

    def compute_cost(self, X, y):
        """Calculate Mean Squared Error cost."""
        m = len(y)
        predictions = self._hypothesis(X)
        cost = np.sum((predictions - y) ** 2) / (2 * m)
        return cost

    def fit(self, X, y, learning_rate=None, num_iterations=None):
        """Train the model using gradient descent."""
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if num_iterations is not None:
            self.num_iterations = num_iterations
            
        X = self._add_intercept(X)
        m, n = X.shape
        self.theta = np.zeros((n, 1))
        self.cost_history = []
        
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        for i in range(self.num_iterations):
            predictions = self._hypothesis(X)
            error = predictions - y
            
            # Gradient Descent update
            self.theta -= (self.learning_rate / m) * np.dot(X.T, error)

            # Track the cost for each iteration
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

        return self.theta, self.cost_history

    def predict(self, X):
        """Make predictions using the trained model."""
        X = self._add_intercept(X)
        return self._hypothesis(X)

    def tune(self, X_train, y_train, X_val, y_val, learning_rates, iterations):
        """Tune the model to find the best learning rate from a list of options."""
        best_learning_rate = None
        best_cost = float('inf')
        
        for lr in learning_rates:
            self.fit(X_train, y_train, learning_rate=lr, num_iterations=iterations)
            val_cost = self.compute_cost(self._add_intercept(X_val), y_val)
            
            print(f"Learning rate: {lr}, Validation Cost: {val_cost:.4f}")

            if val_cost < best_cost:
                best_cost = val_cost
                best_learning_rate = lr

        print(f"Best learning rate found: {best_learning_rate} with Validation Cost: {best_cost:.4f}")
        return best_learning_rate, best_cost



# Initialize and train the Linear Regression model without tuning
linear_model = LinearRegression(learning_rate=1e-3, num_iterations=100)
linear_model.fit(X_train_scaled, Y_train.to_numpy().reshape(-1, 1))

# Predictions on the test set
Y_pred_test_lr_before = linear_model.predict(X_test_scaled)

# Tuning the Linear Regression Model
best_learning_rate, _ = linear_model.tune(
    X_train_scaled, Y_train.to_numpy().reshape(-1, 1),
    X_val_scaled, Y_val.to_numpy().reshape(-1, 1),
    [1e-7, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2], iterations=1000
)

# Set the best learning rate for the tuned model
linear_model.learning_rate = best_learning_rate

# Train Linear Regression Model with tuned learning rate
linear_model.num_iterations = 1000
linear_model.fit(X_train_scaled, Y_train.to_numpy().reshape(-1, 1))


# Predictions on the test set after tuning
Y_pred_test_lr_after = linear_model.predict(X_test_scaled)

# Calculate metrics for Linear Regression before tuning
mse_test_lr_before = mean_squared_error(Y_test, Y_pred_test_lr_before)
mae_test_lr_before = mean_absolute_error(Y_test, Y_pred_test_lr_before)
r2_test_lr_before = r2_score(Y_test, Y_pred_test_lr_before)

# Calculate metrics for Linear Regression after tuning
mse_test_lr_after = mean_squared_error(Y_test, Y_pred_test_lr_after)
mae_test_lr_after = mean_absolute_error(Y_test, Y_pred_test_lr_after)
r2_test_lr_after = r2_score(Y_test, Y_pred_test_lr_after)

# Print performance metrics for Linear Regression before tuning
print("Linear Regression Performance Metrics (Before Tuning):")
print(f"Test MSE: {mse_test_lr_before:.2f}")
print(f"Test MAE: {mae_test_lr_before:.2f}")
print(f"Test R²: {r2_test_lr_before:.2f}")

# Print performance metrics after tuning
print("\nLinear Regression Performance Metrics (After Tuning):")
print(f"Test MSE: {mse_test_lr_after:.2f}")
print(f"Test MAE: {mae_test_lr_after:.2f}")
print(f"Test R²: {r2_test_lr_after:.2f}")







