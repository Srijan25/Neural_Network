import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Load the emoji data from the Excel file
data = pd.read_csv('emoji_data.csv', sep=',', header=None, names=['Emoji', 'Label'])

# Extract emoji data and labels
emoji_data = data['Emoji'].values.reshape(-1, 1).astype(np.float32)
emoji_labels = data['Label'].values.astype(np.float32)

# Create a sequential neural network model
model = Sequential()

# Add layers to the model
model.add(Dense(units=4, activation='relu', input_dim=1))  # Input layer
model.add(Dense(units=2, activation='relu'))              # Hidden layer
model.add(Dense(units=1, activation='tanh'))              # Output layer

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(emoji_data, emoji_labels, epochs=1000, verbose=1)

# Test the model
test_data = np.array([[105]])  # Test data (replace with your data)
predictions = model.predict(test_data)

if predictions[0][0] > 0:
    print("Positive emoji")
else:
    print("Negative emoji")
