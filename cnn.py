import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.regularizers import l2


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Load the JSON data
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def normalize_sample(sample):
    sample = np.array(sample)
    mean_val = sample.mean()
    std_val = sample.std()

    # Avoid division by zero
    if std_val == 0:
        return np.zeros_like(sample)

    normalized_sample = (sample - mean_val) / std_val
    return normalized_sample


# Preprocess the data with per-sample normalization
def preprocess_data(data):
    X = []
    y = []

    # Extract features and labels
    for entry in data:
        X.append(entry['data'])
        y.append(entry['label'])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Normalize each sample individually
    X_normalized = np.array([normalize_sample(sample) for sample in X])

    # Add a channel dimension for Conv1D
    X_normalized = X_normalized[..., np.newaxis]  # Shape becomes (samples, time_steps, 1)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Converts to integers (0, 1, 2)
    y_one_hot = to_categorical(y_encoded, num_classes=3)  # One-hot encode the labels

    return X_normalized, y_one_hot, label_encoder

# Update Model Input Shape
def build_model(input_shape):
    model = tf.keras.Sequential([
        # 1D Convolutional Layer
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Flatten and Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        tf.keras.layers.Dense(3, activation='softmax')  # Output layer for 3 classes
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def add_noise(X, noise_factor=0.05):
    std_devs = np.std(X, axis=1, keepdims=True)
    
    std_devs = np.where(std_devs == 0, 1, std_devs)
    
    noise = noise_factor * std_devs * np.random.randn(*X.shape)
    
    return X + noise


# Main code
if __name__ == "__main__":
    # Load and preprocess data
    json_file = "eeg_data.json"  # Replace with your JSON file path
    data = load_data(json_file)
    X, y, label_encoder = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add noise for data augmentation
    X_augmented = add_noise(X_train)
    X_train = np.concatenate((X_train, X_augmented))  # Combine original and augmented data
    y_train = np.concatenate((y_train, y_train))      # Duplicate labels for augmented data

    # Build the model
    input_shape = X_train.shape[1:]  # (500, 1)
    model = build_model(input_shape)

    # Train the model
    callback = tf.keras.callbacks.EarlyStopping(patience=8)
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[callback])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save("eye_classification_model.h5")

    # Inverse transform predictions to get class labels
    y_pred = model.predict(X_test)
    y_pred_classes = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))

    # Plot the training history
    plot_history(history)

    y_test_classes = label_encoder.inverse_transform(np.argmax(y_test, axis=1))  # True labels
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap='viridis', xticks_rotation='vertical')

    plt.title("Confusion Matrix")
    plt.show()
