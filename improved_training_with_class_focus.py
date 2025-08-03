import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("🚀 Starting IMPROVED Training with Class Focus")
print("=" * 50)

# Data loading
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    'val',
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42
)

class_names = training_set.class_names
print(f"📊 Classes: {class_names}")

# Calculate class distribution and weights
def calculate_class_weights(dataset):
    """Calculate class weights to handle imbalanced data"""
    y_true = []
    for _, labels in dataset:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_true),
        y=y_true
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("📊 Class distribution:")
    unique, counts = np.unique(y_true, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"   {class_names[class_idx]}: {count} samples (weight: {class_weights[i]:.2f})")
    
    return class_weight_dict

class_weights = calculate_class_weights(training_set)

# Enhanced data augmentation - especially for rust classes
def create_rust_focused_augmentation():
    """Create augmentation that helps distinguish between rust types"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),  # More rotation for rust patterns
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.3),   # More contrast to highlight rust colors
        tf.keras.layers.RandomBrightness(0.2),
        # Add some color jittering to help with rust color differences
        tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, 0.1)),
        tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, 0.8, 1.2)),
    ])

data_augmentation = create_rust_focused_augmentation()

# Preprocessing with augmentation
def preprocess_data(dataset, augment=False):
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

training_set = preprocess_data(training_set, augment=True)
validation_set = preprocess_data(validation_set, augment=False)

# Enhanced model with focus on feature discrimination
def create_enhanced_model():
    """Create model with better feature extraction for similar classes"""
    model = Sequential([
        Input(shape=(128, 128, 3)),
        
        # First block - basic features
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second block - texture features
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third block - pattern features
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth block - complex features
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fifth block - high-level features (helps with similar classes)
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        
        # Classification head with more capacity
        Dropout(0.5),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(len(class_names), activation='softmax')
    ])
    
    return model

model = create_enhanced_model()

# Compile with class weights consideration
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Slightly lower LR for stability
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_2_accuracy']
)

print("📋 Enhanced Model Summary:")
model.summary()

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # More patience
        restore_best_weights=True,
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive LR reduction
        patience=7,
        min_lr=1e-8,
        verbose=1
    ),
    
    ModelCheckpoint(
        'best_rust_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("🎯 Starting enhanced training...")
print("⚠️ This will take longer but should better distinguish rust types!")

# Train with class weights
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=60,  # More epochs
    callbacks=callbacks,
    class_weight=class_weights,  # This helps with imbalanced classes
    verbose=1
)

print("✅ Enhanced training completed!")

# Evaluate
train_loss, train_acc, train_top2 = model.evaluate(training_set, verbose=0)
val_loss, val_acc, val_top2 = model.evaluate(validation_set, verbose=0)

print(f"📈 Training Accuracy: {train_acc:.4f}")
print(f"📈 Validation Accuracy: {val_acc:.4f}")

# Save the model
model.save("trained_model.keras")
print("💾 Enhanced model saved as 'trained_model.keras'")

# Test specifically on rust classes
print("\n🧪 Testing rust class discrimination...")
try:
    for images, labels in validation_set.take(1):
        for i in range(min(5, len(images))):
            prediction = model.predict(images[i:i+1], verbose=0)
            predicted_class = class_names[np.argmax(prediction[0])]
            actual_class = class_names[np.argmax(labels[i])]
            
            if 'rust' in actual_class.lower():
                yellow_rust_conf = prediction[0][2] * 100  # Yellow Rust index
                brown_rust_conf = prediction[0][3] * 100   # brown rust index
                
                print(f"  Actual: {actual_class}")
                print(f"  Predicted: {predicted_class}")
                print(f"  Yellow Rust: {yellow_rust_conf:.1f}% | Brown Rust: {brown_rust_conf:.1f}%")
                print("  ---")
except Exception as e:
    print(f"⚠️ Test failed: {e}")

print("\n🎉 ENHANCED TRAINING COMPLETE!")
print("📋 Copy 'trained_model.keras' to your backend and test!")
