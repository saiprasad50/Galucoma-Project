import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# --- Configuration (Adjust if needed) ---
base_dir = 'data'  # Assuming your script is run from the 'ss' folder
img_width, img_height = 224, 224 # VGG16 default input size
batch_size = 16 # Use a smaller batch size for small datasets
epochs = 20 # Increase epochs since we are training only a small part of the model
model_filename = 'glaucoma_vgg16_transfer_model.h5'

# Define the paths
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# --- 1. Enhanced Data Augmentation ---
# Stronger augmentation to compensate for small dataset size and reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Rotate up to 30 degrees
    width_shift_range=0.3,  # Shift horizontal position
    height_shift_range=0.3, # Shift vertical position
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,   # Critical for generalizability
    vertical_flip=True,     # Useful for fundus images
    brightness_range=[0.7, 1.3], # Adjust brightness
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary' 
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# --- 2. Load Pre-trained VGG16 Base Model ---
conv_base = VGG16(
    weights='imagenet',          # Use weights pre-trained on ImageNet
    include_top=False,           # Exclude the classifier layers at the top
    input_shape=(img_width, img_height, 3)
)

# Freeze the convolutional base to prevent weights from being updated
conv_base.trainable = False

# --- 3. Build the New Model (VGG16 + Custom Classifier) ---
model = Sequential()
model.add(conv_base) # Add the frozen VGG16 feature extractor

# Add your custom classifier layers on top
model.add(Flatten())
model.add(BatchNormalization()) # Helps training deep networks
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) # High dropout for strong regularization
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid')) # Binary classification output

# --- 4. Compile and Train ---
# Use a lower learning rate for stability when dealing with small data
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4), # Lower learning rate
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# --- Save the Model ---
model.save(model_filename)
print(f"\nModel saved successfully as: {model_filename}")

# Class indices (for Flask app)
print("\nClass Indices:")
print(train_generator.class_indices)