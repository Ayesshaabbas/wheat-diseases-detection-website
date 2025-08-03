import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("🚀 Starting Improved Wheat Disease Training")
print("=" * 50)

# Check if GPU is available
print(f"🔧 TensorFlow version: {tf.__version__}")
print(f"🔧 GPU available: {tf.config.list_physical_devices('GPU')}")

# Data loading with improved augmentation
print("📂 Loading training data...")
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',  # Make sure this path is correct
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42
)

print("📂 Loading validation data...")
validation_set = tf.keras.utils.image_dataset_from_directory(
    'val',  # Make sure this path is correct
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42
)

# Get class names
class_names = training_set.class_names
print(f"📊 Classes found: {class_names}")
print(f"📊 Number of classes: {len(class_names)}")

# Check dataset sizes
train_size = tf.data.experimental.cardinality(training_set).numpy()
val_size = tf.data.experimental.cardinality(validation_set).numpy()
print(f"📊 Training batches: {train_size}")
print(f"📊 Validation batches: {val_size}")

# Data augmentation - CRITICAL for better performance
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

# Apply augmentation and preprocessing
def preprocess_data(dataset, augment=False):
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    # Normalize to [0,1] - CRITICAL
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    # Performance optimization
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

print("🔄 Applying data preprocessing...")
training_set = preprocess_data(training_set, augment=True)
validation_set = preprocess_data(validation_set, augment=False)

# Improved model architecture - simpler but more effective
print("🏗️ Building improved model...")
model = Sequential([
    Input(shape=(128, 128, 3)),
    
    # First block
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Second block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Third block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Fourth block
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    GlobalAveragePooling2D(),  # Better than Flatten
    
    # Classification head
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Improved compilation with better learning rate
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Higher learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_2_accuracy']
)

print("📋 Model Summary:")
model.summary()

# Callbacks for better training
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("🎯 Starting training with improved settings...")
print("⏱️ This will take longer but produce much better results!")

# Train for more epochs with callbacks
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=50,  # More epochs with early stopping
    callbacks=callbacks,
    verbose=1
)

print("✅ Training completed!")

# Evaluate the model
print("📊 Evaluating model...")
train_loss, train_acc, train_top2 = model.evaluate(training_set, verbose=0)
val_loss, val_acc, val_top2 = model.evaluate(validation_set, verbose=0)

print(f"📈 Training Accuracy: {train_acc:.4f}")
print(f"📈 Validation Accuracy: {val_acc:.4f}")
print(f"📈 Training Top-2 Accuracy: {train_top2:.4f}")
print(f"📈 Validation Top-2 Accuracy: {val_top2:.4f}")

# Save the final model
model.save("trained_model.keras")
print("💾 Model saved as 'trained_model.keras'")

# Save training history
import json
with open("training_history.json", "w") as f:
    # Convert numpy types to regular Python types for JSON serialization
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    json.dump(history_dict, f)

print("💾 Training history saved")

# Plot training results
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Learning rate plot (if available)
plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning Rate', color='green')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
else:
    plt.text(0.5, 0.5, 'Learning Rate\nNot Recorded', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12)
    plt.title('Learning Rate')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Training plots saved as 'training_results.png'")

# Test the model with a sample prediction
print("🧪 Testing model with sample prediction...")
try:
    # Get a sample from validation set
    for images, labels in validation_set.take(1):
        sample_image = images[0:1]  # Take first image
        sample_label = labels[0:1]  # Take first label
        
        prediction = model.predict(sample_image, verbose=0)
        predicted_class = class_names[np.argmax(prediction[0])]
        actual_class = class_names[np.argmax(sample_label[0])]
        confidence = np.max(prediction[0]) * 100
        
        print(f"🎯 Sample prediction:")
        print(f"   Predicted: {predicted_class} ({confidence:.2f}%)")
        print(f"   Actual: {actual_class}")
        print(f"   All predictions: {[f'{class_names[i]}: {prediction[0][i]*100:.1f}%' for i in range(len(class_names))]}")
        break
except Exception as e:
    print(f"⚠️ Sample prediction failed: {e}")

print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print("=" * 50)
print("📋 Next steps:")
print("1. Copy 'trained_model.keras' to your backend directory")
print("2. Restart your Flask backend")
print("3. Test with the web interface")
print("=" * 50)
