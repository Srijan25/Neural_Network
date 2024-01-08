import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate a sine wave time series data
# t = np.linspace(0, 100, 500)
# sin_wave = np.sin(t) + np.random.normal(scale=0.3, size=len(t))

# Load the data
data = pd.read_csv('data.csv', sep=',', header=None, names=['Time', 'Value'])


# Plot the data
plt.figure(figsize=(12, 6))
plt.title("Generated Sine Wave Time Series Data")
# plt.plot(t, sin_wave, label='Sine Wave + Noise')

plt.plot(data['Time'], data['Value'], label='Sine Wave + Noise')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Prepare the data for training the RNN
window_size = 10

X, y = [], []
# for i in range(len(sin_wave) - window_size):
#     X.append(sin_wave[i:i + window_size])
#     y.append(sin_wave[i + window_size])

for i in range(len(data['Value']) - window_size):
    X.append(data['Value'][i:i + window_size])
    y.append(data['Value'][i + window_size])


X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for RNN input

model = Sequential()

# RNN layer with 50 units
model.add(SimpleRNN(units=50, activation='relu', input_shape=(window_size, 1)))

# Output layer
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=50, batch_size=32)

# Generate new sequences for prediction
# test_data = sin_wave[-window_size:]
test_data = data['Value'][-window_size:]
test_data = np.reshape(test_data, (1, window_size, 1))

# Predict next 50 values
predicted_values = []
for _ in range(50):
    prediction = model.predict(test_data)
    predicted_values.append(prediction[0, 0])
    test_data = np.roll(test_data, -1)  # Shift the window by one step
    test_data[0, -1, 0] = prediction[0, 0]  # Add the predicted value at the end of the window

# Plot the original data and predicted values
plt.figure(figsize=(12, 6))
plt.title("Sine Wave Time Series Prediction using RNN")
# plt.plot(t, sin_wave, label='Original Data')
# plt.plot(np.arange(len(sin_wave), len(sin_wave) + len(predicted_values)), predicted_values, label='Predicted Data')

plt.plot(data['Time'], data['Value'], label='Original Data')
plt.plot(np.arange(len(data['Value']), len(data['Value']) + len(predicted_values)), predicted_values, label='Predicted Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()