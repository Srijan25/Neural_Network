import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Generate sample data
np.random.seed(0)
num_samples = 1000
num_features = 2

X = np.random.rand(num_samples, num_features) * 100  # Random features (e.g., bedrooms, square footage)
y = 0.5 * X[:, 0] + 0.8 * X[:, 1] + 10 + np.random.randn(num_samples) * 5  # Generate house prices (with some noise)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(10, input_dim=num_features, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
batch_size = 32
epochs = 100
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {loss:.4f}")

# Make predictions
sample_input = np.array([[3, 1500]])  # Sample input with 3 bedrooms and 1500 sq. ft.
sample_input = scaler.transform(sample_input)
predicted_price = model.predict(sample_input)
print(f"Predicted House Price: ${predicted_price[0][0]:.2f}")
