import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, 
                                     Conv2D, MaxPooling2D, Flatten)
from tensorflow.keras import regularizers, layers, models
from IPython.display import Image
import os
import shutil  # For creating/copying files/folders if needed
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import classification_report


# ------------------------------------------------------------------------
# 1) Helper functions vs. image_dataset_from_directory
# ------------------------------------------------------------------------
# NOTE: image_dataset_from_directory by default will simply resize images 
#       to (image_size x image_size) without random aspect ratio changes 
#       or random crops. If you want random resizing/cropping, you DO need 
#       these helper functions.

def resize_and_random_crop(image, final_size=512):
    """
    Resizes the shorter side of `image` to `final_size` px while keeping aspect ratio,
    then performs a random crop to (final_size, final_size).
    """
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    
    shorter_side = tf.minimum(height, width)
    scale_ratio = tf.cast(final_size, tf.float32) / tf.cast(shorter_side, tf.float32)
    
    new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * scale_ratio), tf.int32)
    new_width  = tf.cast(tf.round(tf.cast(width,  tf.float32) * scale_ratio), tf.int32)
    
    # Resize while keeping aspect ratio
    image = tf.image.resize(image, (new_height, new_width))
    
    # Random Crop
    cropped_image = tf.image.random_crop(image, size=[final_size, final_size, 3])
    cropped_image = cropped_image / 255.0  # scale to [0,1]
    
    return cropped_image


def resize_and_random_crop_single(img, final_size=512):
    """Handles a single image of shape [H, W, 3]."""
    img = tf.cast(img, tf.float32)
    height, width = tf.shape(img)[0], tf.shape(img)[1]
    shorter_side = tf.minimum(height, width)
    scale_ratio = tf.cast(final_size, tf.float32) / tf.cast(shorter_side, tf.float32)

    new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * scale_ratio), tf.int32)
    new_width  = tf.cast(tf.round(tf.cast(width,  tf.float32) * scale_ratio), tf.int32)

    img = tf.image.resize(img, (new_height, new_width))
    img = tf.image.random_crop(img, size=[final_size, final_size, 3])
    img /= 255.0
    return img


def preprocess_train(batch_images, batch_labels, final_size=512):
    """
    Apply random crop to each image using map_fn.
    """
    batch_images = tf.map_fn(
        lambda img: resize_and_random_crop_single(img, final_size),
        batch_images
    )
    return batch_images, batch_labels


def preprocess_val(batch_images, batch_labels, final_size=512):
    """
    (Could do center crop for validation, but for simplicity using the same random crop)
    """
    batch_images = tf.map_fn(
        lambda img: resize_and_random_crop_single(img, final_size),
        batch_images
    )
    return batch_images, batch_labels


# ------------------------------------------------------------------------
# 2) Loading Datasets
# ------------------------------------------------------------------------
dataset_dir = 'Training Images'
validation_dir = 'Testing Images'

print("Loading dataset from:", dataset_dir)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

batch_size = 8

# Main train dataset
train_ds1 = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    label_mode='int',        # 0 or 1 labels
    batch_size=batch_size,
    image_size=(256, 256),
    seed=512
)

# Additional train dataset
train_ds2 = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, 'train_potato'),
    label_mode='int',
    batch_size=batch_size,
    image_size=(256, 256),
    seed=512
)

train_ds2_count = tf.data.experimental.cardinality(train_ds2).numpy()
split_80 = int(0.8 * train_ds2_count)

train_ds2_train = train_ds2.take(split_80)
train_ds2_val   = train_ds2.skip(split_80)

# Check class names
class_names = train_ds1.class_names  # This should reflect the subfolders. 
print("Training Classes (from train_ds1):", class_names)

# Concatenate the train sets
train_ds = train_ds1.concatenate(train_ds2_train)

# Validation dataset(s)
val_ds_main = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    label_mode='int',
    batch_size=batch_size,
    image_size=(256, 256),
    seed=512
)
val_ds = val_ds_main.concatenate(train_ds2_val)

# ------------------------------------------------------------------------
# 3) Preprocessing
# ------------------------------------------------------------------------


train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds   = val_ds.map(lambda x, y: (x/255.0, y))


#AUTOTUNE = tf.data.AUTOTUNE

#train_ds = (train_ds
#            .map(preprocess_train, num_parallel_calls=AUTOTUNE)
#            .prefetch(AUTOTUNE))

#val_ds = (val_ds
#          .map(preprocess_val, num_parallel_calls=AUTOTUNE)
#          .prefetch(AUTOTUNE))

# ------------------------------------------------------------------------
# 4) Simple CNN (Binary Classification)
# ------------------------------------------------------------------------
model = models.Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2,2)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # single output for binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------------------
# 5) Train the model
# ------------------------------------------------------------------------
checkpoint_cb = ModelCheckpoint(
    'best_model_test1.h5',    # path to .h5 file 
    monitor='val_loss',
    save_best_only=True
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[checkpoint_cb]
)

# ------------------------------------------------------------------------
# 6) Evaluate on the validation set
# ------------------------------------------------------------------------
val_loss, val_acc = model.evaluate(val_ds)
print("\nValidation Loss: {:.4f}".format(val_loss))
print("Validation Accuracy: {:.2f}%".format(val_acc * 100))

# ------------------------------------------------------------------------
# 7) Predictions and Misidentified Images
# ------------------------------------------------------------------------
y_true = []
y_pred = []

# Create folders for misidentified images
misidentified_real_folder = 'Misidentified Real Images'
misidentified_fake_folder = 'Misidentified Fake Images'
os.makedirs(misidentified_real_folder, exist_ok=True)
os.makedirs(misidentified_fake_folder, exist_ok=True)

misidentified_fake_count = 0  # to limit saving to first 100
misidentified_real_count = 0

for images_batch, labels_batch in val_ds:
    # predictions
    preds = model.predict(images_batch)
    pred_classes = (preds > 0.5).astype(int).flatten()
    true_classes = labels_batch.numpy().astype(int)

    # record for confusion matrix & classification report
    y_true.append(true_classes)
    y_pred.append(pred_classes)

    # Save misidentified images:
    #  - Real images (label=0) predicted as Fake (pred=1) => to 'Misidentified Real Images'
    #  - Fake images (label=1) predicted as Real (pred=0) => to 'Misidentified Fake Images' (only first 100)
    for i in range(len(images_batch)):
        if true_classes[i] == 0 and pred_classes[i] == 1 and misidentified_real_count < 100:
            # Real identified as Fake
            misidentified_real_count += 1
            img_array = images_batch[i].numpy()
            img_array = (img_array * 255).astype(np.uint8)  # scale back to 0-255
            filename = os.path.join(misidentified_real_folder, f'real_as_fake_{np.random.randint(1e6)}.jpg')
            tf.keras.preprocessing.image.save_img(filename, img_array)

        elif true_classes[i] == 1 and pred_classes[i] == 0 and misidentified_fake_count < 100:
            # Fake identified as Real (limit to 100 images)
            misidentified_fake_count += 1
            img_array = images_batch[i].numpy()
            img_array = (img_array * 255).astype(np.uint8)
            filename = os.path.join(misidentified_fake_folder, f'fake_as_real_{np.random.randint(1e6)}.jpg')
            tf.keras.preprocessing.image.save_img(filename, img_array)

y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)

# ------------------------------------------------------------------------
# 8) Confusion Matrix & Classification Report
# ------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# Ensure the model is assigning class names consistently:
#   Typically, if class_names=['REAL','FAKE'], then label 0->REAL, 1->FAKE.
target_names = ['FAKE', 'REAL']

# Instead of a technical classification report, let's make it more readable:
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
print("\n===== Classification Report =====")
print("Below is a performance breakdown for each class:")
print(report)

# ------------------------------------------------------------------------
# (Optional) Save the model as a .keras file
# ------------------------------------------------------------------------
model.save('final_model_256.keras')  # saves in Keras (SavedModel) format


# ------------------------------------------------------------------------
# (Optional) Save the Classification Report
# ------------------------------------------------------------------------

report_dict = classification_report(
    y_true, 
    y_pred, 
    target_names=target_names, 
    digits=4, 
    output_dict=True
)

# Convert the dictionary to a DataFrame
df_report = pd.DataFrame(report_dict).transpose()

# Create a nice text-based table
report_table = tabulate(df_report, headers='keys', tablefmt='psql')

# Print to console (optional)
print(report_table)

# Save to a text file
with open('classification_report_formatted.txt', 'w') as f:
    f.write(report_table)

# ------------------------------------------------------------------------
# 9) Plot training & validation metrics
# ------------------------------------------------------------------------

history_dict = history.history

train_acc = history_dict['accuracy']
train_loss = history_dict['loss']
val_acc = history_dict.get('val_accuracy')
val_loss = history_dict.get('val_loss')

epochs = range(1, len(train_acc) + 1)

# Create a new figure
plt.figure(figsize=(12, 5))

# Subplot for Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
if val_acc is not None:
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Subplot for Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
if val_loss is not None:
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the figure to a file (e.g., PNG, JPG, PDF, etc.)
plt.savefig("training_validation_metrics_lowerResolution.png", dpi=300)  # adjust dpi as needed

# Optional: Display the figure
plt.show()
