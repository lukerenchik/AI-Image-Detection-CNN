import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ============================================================================
# 1) Directory Structure & Data Loading
# ----------------------------------------------------------------------------
# Recommended folder structure:
#
# dataset/
#    train/
#         Human/   (or "Real")
#         AI/      (or "Fake")
#    validation/ (or "test/")
#         Human/
#         AI/
#
# Set these paths to the folders you’ve created:
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/validation"

# Set some parameters
IMG_SIZE = 224     # EfficientNetB0 default input size (you can also try 256)
BATCH_SIZE = 32    # Adjust based on your GPU/CPU memory

# Load datasets using image_dataset_from_directory:
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",          # integer labels (0, 1)
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False
)

# Get the class names (make sure they are ordered as desired)
class_names = train_ds.class_names
print("Detected classes:", class_names)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

# ============================================================================
# 2) Data Augmentation & Preprocessing Layers
# ----------------------------------------------------------------------------
# These layers are added at the beginning of the model.
data_augmentation = models.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Note: EfficientNetB0 expects inputs preprocessed with its dedicated function.
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# ============================================================================
# 3) Building the Model using Transfer Learning (EfficientNetB0)
# ----------------------------------------------------------------------------
# Load the EfficientNetB0 model with pretrained ImageNet weights.
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze the base model

# Create the model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# Apply data augmentation
x = data_augmentation(inputs)
# Preprocess for EfficientNet
x = preprocess_input(x)
# Pass through the base model
x = base_model(x, training=False)
# Global pooling layer
x = layers.GlobalAveragePooling2D()(x)
# Optional dropout for regularization
x = layers.Dropout(0.3)(x)
# Final output: 1 neuron with sigmoid activation for binary classification
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================================
# 4) Callbacks and Training
# ----------------------------------------------------------------------------
# Save the best model (based on validation loss)
checkpoint_cb = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
# Early stopping to avoid overfitting
early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train the model
EPOCHS = 20  # Increase if you have a large dataset
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# ============================================================================
# 5) Evaluation and Metrics
# ----------------------------------------------------------------------------
# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_ds)
print("\nValidation Loss: {:.4f}".format(val_loss))
print("Validation Accuracy: {:.2f}%".format(val_acc * 100))

# Gather predictions and true labels for detailed metrics
y_true = []
y_pred = []

for images_batch, labels_batch in val_ds:
    preds = model.predict(images_batch)
    pred_classes = (preds > 0.5).astype(int).flatten()
    y_true.extend(labels_batch.numpy())
    y_pred.extend(pred_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# Create a classification report.
# Make sure the order of target names corresponds to the order of labels.
# For example, if class_names = ['AI', 'Human'], then label 0 corresponds to 'AI'
target_names = class_names  
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
print("\n===== Classification Report =====")
print(report)

# Optionally, save the classification report to a text file.
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=target_names,
    digits=4,
    output_dict=True
)
df_report = pd.DataFrame(report_dict).transpose()
report_table = tabulate(df_report, headers="keys", tablefmt="psql")
with open("classification_report_formatted.txt", "w") as f:
    f.write(report_table)

# ============================================================================
# 6) Plot Training and Validation Metrics
# ----------------------------------------------------------------------------
history_dict = history.history
epochs_range = range(1, len(history_dict["accuracy"]) + 1)

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history_dict["accuracy"], "bo-", label="Training Accuracy")
plt.plot(epochs_range, history_dict["val_accuracy"], "ro-", label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history_dict["loss"], "bo-", label="Training Loss")
plt.plot(epochs_range, history_dict["val_loss"], "ro-", label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_validation_metrics.png", dpi=300)
plt.show()

# ============================================================================
# 7) (Optional) Saving the Final Model
# ----------------------------------------------------------------------------
model.save("final_model_224.keras")
