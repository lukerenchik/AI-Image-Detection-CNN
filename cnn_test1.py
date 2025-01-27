import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import regularizers, layers, models
from IPython.display import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------------------------------------------------------------
# 1) Define helper function: resize_and_random_crop
# ------------------------------------------------------------------------
def resize_and_random_crop(image, final_size=512):
    """
    Resizes the shorter side of `image` to `final_size` px while keeping aspect ratio,
    then performs a random crop to (final_size, final_size).
    """
    # Convert image to float32 if needed
    image = tf.cast(image, tf.float32)

    # 1) Compute the current height and width
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    
    # 2) Calculate the scaling ratio
    shorter_side = tf.minimum(height, width)
    scale_ratio = tf.cast(final_size, tf.float32) / tf.cast(shorter_side, tf.float32)
    
    # 3) Compute new target size
    new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * scale_ratio), tf.int32)
    new_width  = tf.cast(tf.round(tf.cast(width, tf.float32) * scale_ratio), tf.int32)
    
    # 4) Resize (this maintains aspect ratio)
    image = tf.image.resize(image, (new_height, new_width))
    
    # 5) Random Crop to final_size x final_size
    cropped_image = tf.image.random_crop(image, size=[final_size, final_size, 3])
    
    # (Optionally) Scale pixel values to [0,1]
    cropped_image = cropped_image / 255.0
    
    return cropped_image

def resize_and_center_crop_single(img, final_size=512):
    """
    Resizes the shorter side of `img` to `final_size` px (maintaining aspect ratio),
    then does a centered (final_size, final_size) crop.
    """
    # Convert to float32
    img = tf.cast(img, tf.float32)
    
    # Current image shape
    height, width = tf.shape(img)[0], tf.shape(img)[1]
    
    # Compute scale ratio based on the shorter side
    shorter_side = tf.minimum(height, width)
    scale_ratio = tf.cast(final_size, tf.float32) / tf.cast(shorter_side, tf.float32)
    
    # Calculate new dimensions
    new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * scale_ratio), tf.int32)
    new_width  = tf.cast(tf.round(tf.cast(width,  tf.float32) * scale_ratio), tf.int32)
    
    # Resize
    img = tf.image.resize(img, (new_height, new_width))
    
    # Now compute offsets to extract a centered final_size x final_size patch
    offset_height = (new_height - final_size) // 2
    offset_width  = (new_width - final_size) // 2
    
    # Crop to center
    img = tf.image.crop_to_bounding_box(
        img,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=final_size,
        target_width=final_size
    )
    
    # Scale pixel values to [0,1]
    img /= 255.0
    
    return img


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
    batch_images shape: [batch_size, height, width, 3]
    Apply random crop to each image using map_fn.
    """
    batch_images = tf.map_fn(
        lambda img: resize_and_random_crop_single(img, final_size),
        batch_images
    )
    return batch_images, batch_labels

def preprocess_val(batch_images, batch_labels, final_size=512):
    batch_images = tf.map_fn(
        lambda img: resize_and_center_crop_single(img, final_size),
        batch_images
    )
    return batch_images, batch_labels




# ------------------------------------------------------------------------
# 2) Set directories & load datasets
# ------------------------------------------------------------------------
dataset_dir = 'Training Images'
validation_dir = 'Testing Images'

print("Loading dataset from: ", dataset_dir)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

batch_size = 16

# train_ds1
train_ds1 = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    label_mode='int',
    batch_size=batch_size,
    image_size=(512,512),  # Keep original size
    seed=512
)

# train_ds2 - we will split this 80/20
train_ds2 = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, 'train_potato'),
    label_mode='int',
    batch_size=batch_size,
    image_size=None,
    seed=512
)

# Step 2a: Split train_ds2 into 80% train, 20% validation
train_ds2_count = tf.data.experimental.cardinality(train_ds2).numpy()
split_80 = int(0.8 * train_ds2_count)

train_ds2_train = train_ds2.take(split_80)
train_ds2_val   = train_ds2.skip(split_80)

class_names = train_ds1.class_names  # or train_ds2.class_names
train_ds = train_ds1.concatenate(train_ds2)




# val_ds: main validation from the Testing Images dir
val_ds_main = tf.keras.utils.image_dataset_from_directory(
    os.path.join(validation_dir),
    label_mode='int',
    batch_size=batch_size,
    image_size=(512,512),
    seed=512
)

# (Optionally) Combine val_ds_main with the new val split from train_ds2
# If you want a single big validation dataset:
val_ds = val_ds_main.concatenate(train_ds2_val)

# If you prefer to keep them separate, just keep them as two distinct sets and evaluate both.

print("Training Classes: ", class_names)


# ------------------------------------------------------------------------
# 3) Map the preprocessing to each dataset
# ------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (train_ds
    .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (val_ds
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# ------------------------------------------------------------------------
# 4) Define a simple CNN
# ------------------------------------------------------------------------
num_classes = 2  # [REAL, FAKE]

model = models.Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(512, 512, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (5,5), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (5,5), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------------------
# 5) Train the model
# ------------------------------------------------------------------------
checkpoint_cb = ModelCheckpoint(
    'best_model_test1.h5',       # file path
    monitor='val_loss',    # metric to monitor
    save_best_only=True    # only save if val_loss improves
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)

# ------------------------------------------------------------------------
# 6) Evaluate on the validation set
# ------------------------------------------------------------------------
val_loss, val_acc = model.evaluate(val_ds)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    # Labels are integers (0 or 1), so no need for argmax
    true_classes = labels.numpy().astype(int)
    # Predicted probabilities need to be thresholded at 0.5 to classify as 0 or 1
    pred_classes = (preds > 0.5).astype(int).flatten()

    y_true.append(true_classes)
    y_pred.append(pred_classes)

y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

target_names = ['REAL', 'FAKE']
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

# Save the model in the root folder
model.save('best_model_test1_root.keras')