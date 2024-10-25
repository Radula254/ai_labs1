import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset/nairobi_office_prices.csv')

# Use the correct column names from the dataset (SIZE and PRICE)
X = data['SIZE'].values  # Feature (office size)
y = data['PRICE'].values  # Target (office price)

# Function to calculate Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to perform Gradient Descent and update weights
def gradient_descent(X, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * X + c
    # Calculate gradients
    dm = (-2 / n) * np.sum(X * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)
    # Update weights
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Parameters
m, c = np.random.rand(), np.random.rand()  # Initialize random slope (m) and intercept (c)
learning_rate = 0.01
epochs = 10

# Training the model for 10 epochs
for epoch in range(epochs):
    m, c = gradient_descent(X, y, m, c, learning_rate)
    y_pred = m * X + c
    error = mean_squared_error(y, y_pred)
    print(f'Epoch {epoch + 1}: Mean Squared Error = {error:.4f}')

# Plot the line of best fit
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, m * X + c, color='red', label='Best fit line')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.legend()
plt.show()

# Predict office price for a size of 100 sq. ft
size = 100
predicted_price = m * size + c
print(f'Predicted office price for size {size} sq. ft: {predicted_price:.2f}')
