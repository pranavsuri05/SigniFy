import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2

# Load dataset
df = pd.read_csv("gesture_data.csv")

# Encode labels (convert text to numbers)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Split data
X = df.iloc[:, 1:].values  # Landmark positions
y = df["label"].values  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for 1D CNN (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the model (1D CNN)
model = keras.Sequential([
    layers.Conv1D(64, 3, activation="relu", input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(label_encoder.classes_), activation="softmax")  # Output layer
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model & label encoder
model.save("sign_language_model.h5")
np.save("label_encoder.npy", label_encoder.classes_)

print("✅ Model training complete & saved!")

# ✅ ADD THIS FUNCTION FOR PREDICTION
def predict_sign(landmarks):
    """
    Predicts the sign language gesture from input landmarks.
    :param landmarks: List or NumPy array of landmark positions (same format as training data).
    :return: Predicted sign label.
    """
    # Load the trained model & encoder
    model = keras.models.load_model("sign_language_model.h5")
    label_classes = np.load("label_encoder.npy", allow_pickle=True)

    # Preprocess input
    landmarks = np.array(landmarks).reshape(1, -1, 1)  # Reshape to match model input

    # Make prediction
    predictions = model.predict(landmarks)
    predicted_class = np.argmax(predictions)  # Get class index

    return label_classes[predicted_class]  # Return the predicted sign label