import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import build_cnn1d

# Configuration
DATA_DIR = "processed_data"
MODEL_PATH = "cnn1d_model.keras"

# Load test data
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Add channel dimension if necessary
if X_test.ndim == 2:
    X_test = X_test[..., np.newaxis]

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int64)

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. Train the model first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Generate predictions
predictions = model.predict(X_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

# Print evaluation metrics
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Noise", "Earthquake"],
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Noise", "Earthquake"],
    yticklabels=["Noise", "Earthquake"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("1D CNN - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()