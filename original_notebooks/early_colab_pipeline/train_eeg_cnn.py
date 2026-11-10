import tensorflow as tf
from tensorflow.keras import layers, models, constraints
import matplotlib.pyplot as plt

print("Step 1: Specialized EEG-CNN Architecture banaya ja raha hai...")

def build_eeg_cnn(channels=64, samples=481, classes=2):
    model = models.Sequential([
        # Block 1: Temporal Convolution (Sirf waqt/frequency ka pattern seekhna)
        layers.Conv2D(8, (1, 64), padding='same', input_shape=(channels, samples, 1), use_bias=False),
        layers.BatchNormalization(),

        # Block 2: Spatial Convolution (Yeh ratta marne se rokey ga - Har sensor ko alag dekhna)
        layers.DepthwiseConv2D((channels, 1), use_bias=False, depth_multiplier=2,
                               depthwise_constraint=constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.5), # 50% neurons off taake ratta na lagay

        # Block 3: Separable Convolution (Features ko combine karna)
        layers.SeparableConv2D(16, (1, 16), use_bias=False, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.5),

        # Block 4: Classification
        layers.Flatten(),
        layers.Dense(classes, activation='softmax', kernel_constraint=constraints.max_norm(0.25))
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Naya model call karna
model = build_eeg_cnn()

print("✅ Architecture Ready! Parameters check karein:")
model.summary()

print("\nStep 2: Model ki Training Shuru (Please wait)...")

# Learning rate ko optimize karne ke liye callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,            # Ziyada epochs kyunke ab ratta nahi lagega
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✅ --- TRAINING COMPLETE --- ✅")

# 📊 Graphs Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.title('EEG-CNN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.title('EEG-CNN Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()