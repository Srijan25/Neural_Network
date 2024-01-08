import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Generate sample data
np.random.seed(0)
num_samples = 300
image_height = 32
image_width = 32
num_channels = 3  # RGB channels

X = np.random.randint(0, 256, (num_samples, image_height, image_width, num_channels), dtype=np.uint8)
y_colors = ['red', 'green', 'blue']
y = np.random.choice(y_colors, num_samples)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode color labels as integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Normalize image data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(image_height, image_width, num_channels)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(y_colors), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 1
epochs = 30
model.fit(X_train, y_train_encoded, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_encoded))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")

# Make predictions by taking input image from disk
image_path = 'blue.jpg'
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_height, image_width))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr = input_arr / 255.0  # Normalize image data
predictions = model.predict(input_arr)
predicted_color = y_colors[np.argmax(predictions[0])]
print(f"Predicted Color: {predicted_color}")


