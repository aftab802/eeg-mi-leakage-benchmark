import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, constraints
import matplotlib.pyplot as plt

print("Step 1: Data ko Normalize (Scale) kar rahe hain (CRITICAL STEP)...")
# Z-Score Normalization: Microvolts ko AI ke samajhne walay standard format me convert karna
mean_val = np.mean(X_train)
std_val = np.std(X_train)

X_train_norm = (X_train - mean_val) / std_val
X_test_norm = (X_test - mean_val) / std_val
print("✅ Normalization Mukammal!")

print("\nStep 2: Official 'EEGNet' Architecture Load kar rahe hain...")
# Industry Standard 'EEGNet' (By Lawhern et al.) specifically for PhysioNet
def build_official_eegnet(nb_classes=2, Chans=64, Samples=481, dropoutRate=0.5):
    model = models.Sequential()

    # Block 1: Temporal
    model.add(layers.Conv2D(8, (1, 80), padding='same', input_shape=(Chans, Samples, 1), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=constraints.max_norm(1.)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling2D((1, 4)))
    model.add(layers.Dropout(dropoutRate))

    # Block 2: Spatial
    model.add(layers.SeparableConv2D(16, (1, 16), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling2D((1, 8)))
    model.add(layers.Dropout(dropoutRate))

    # Classification
    model.add(layers.Flatten())
    model.add(layers.Dense(nb_classes, activation='softmax', kernel_constraint=constraints.max_norm(0.25)))

    return model

model = build_official_eegnet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nStep 3: Normalized Data par Final Training Shuru...")
# Patience thori barha di hai taake model ko seekhne ka poora moqa mile
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)

history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=64, # GPU ka full use karne ke liye batch size double kar diya hai
    validation_data=(X_test_norm, y_test),
    callbacks=[early_stop],
    verbose=1
)

print("\n✅ --- EEGNET TRAINING COMPLETE --- ✅")

# 📊 Final Graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.title('Official EEGNet Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.title('Official EEGNet Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()