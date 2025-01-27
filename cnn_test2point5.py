import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input, BatchNormalization, Add
from tensorflow.keras import regularizers, layers, models
from IPython.display import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model

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

# (Optionally, if you want a *center* crop for validation/test, define a different function.)
# But here we’ll reuse the same random crop for simplicity.
def preprocess_val(batch_images, batch_labels, final_size=512):
    batch_images = tf.map_fn(
        lambda img: resize_and_random_crop_single(img, final_size),
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
    label_mode='categorical',
    class_names=['REAL', 'FAKE'],
    batch_size=batch_size,
    image_size=(512,512),  # Keep original size
    seed=512
)

# train_ds2 - we will split this 80/20
train_ds2 = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, 'train_potato'),
    label_mode='categorical',
    class_names=['REAL', 'FAKE'],
    batch_size=batch_size,
    image_size=(512,512),
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
    label_mode='categorical',
    class_names=['REAL', 'FAKE'],
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
input_layer = Input(shape=(512, 512, 3))


# ---------- Block 1 ----------
x = Conv2D(32, (3, 3), padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# shape is [batch, 256, 256, 32]

# ---------- Block 2 with skip ----------
# transform skip_x to match 64 channels
skip_x = Conv2D(64, (1, 1), padding='same')(x)
skip_x = BatchNormalization()(skip_x)

x2 = Conv2D(64, (3, 3), padding='same')(x)
x2 = BatchNormalization()(x2)
x2 = Conv2D(64, (3, 3), padding='same')(x2)
x2 = BatchNormalization()(x2)

x2 = Add()([skip_x, x2])  # now both branches have shape [..., 64]
x2 = MaxPooling2D((2, 2))(x2)
# shape is [batch, 128, 128, 64]

# ---------- Block 3 with skip ----------
skip_x2 = Conv2D(128, (1, 1), padding='same')(x2)
skip_x2 = BatchNormalization()(skip_x2)

x3 = Conv2D(128, (3, 3), padding='same')(x2)
x3 = BatchNormalization()(x3)
x3 = Conv2D(128, (3, 3), padding='same')(x3)
x3 = BatchNormalization()(x3)

x3 = Add()([skip_x2, x3])  # shape [..., 128]
x3 = MaxPooling2D((2, 2))(x3)
# shape is [batch, 64, 64, 128]

# ---------- Block 4 ----------
# (No skip here or you can do another skip)
x4 = Conv2D(256, (3, 3), padding='same')(x3)
x4 = BatchNormalization()(x4)
x4 = MaxPooling2D((2, 2))(x4)
# shape is [batch, 32, 32, 256]

# ---------- Dense Layers ----------
x_flat = Flatten()(x4)
x_fc = Dense(256, activation='relu')(x_flat)
x_fc = Dropout(0.3)(x_fc)
output_layer = Dense(num_classes, activation='softmax')(x_fc)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------------------
# 5) Train the model
# ------------------------------------------------------------------------
checkpoint_cb = ModelCheckpoint(
    'best_model.h5',       # file path
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
    # Convert one-hot labels to integer class
    true_classes = np.argmax(labels.numpy(), axis=1)
    # Convert predicted probabilities to integer class
    pred_classes = np.argmax(preds, axis=1)

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
model.save('best_model_test2point5_root.h5')