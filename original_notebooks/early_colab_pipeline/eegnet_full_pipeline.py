# Agar MNE delete ho gaya ho toh ensure install ho
!pip install mne -q

import os
import glob
import mne
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, constraints
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc # RAM bachane ke liye

print("Step 1: Pura 109 Subjects ka Data Load, Resample aur Filter ho raha hai (Heavy Process)...")
base_dir = '/content/drive/MyDrive/EEG_Dataset/physionet.org/files/eegmmidb/1.0.0/'
all_files = glob.glob(base_dir + '*/*.edf')

# Sirf Motor Imagery wale runs (4, 8, 12) lenge
target_files = [f for f in all_files if 'R04.edf' in f or 'R08.edf' in f or 'R12.edf' in f]
print(f"Total files to process: {len(target_files)}")

epochs_list = []
for idx, file in enumerate(target_files):
    try:
        raw = mne.io.read_raw_edf(file, preload=True, verbose='ERROR')

        # 🚀 THE FIX: Har file ko strictly 160 Hz par resample kar rahe hain taake crash na ho
        raw.resample(160.0, verbose='ERROR')

        # 🚀 Bandpass Filter (8-30 Hz)
        raw.filter(8., 30., fir_design='firwin', verbose='ERROR')

        events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
        custom_mapping = {}
        if 'T1' in event_dict: custom_mapping['T1'] = 0 # Left Fist
        if 'T2' in event_dict: custom_mapping['T2'] = 1 # Right Fist

        if not custom_mapping: continue
        events, _ = mne.events_from_annotations(raw, event_id=custom_mapping, verbose='ERROR')

        epochs = mne.Epochs(raw, events, event_id=custom_mapping, tmin=0, tmax=3.0,
                            baseline=None, preload=True, verbose='ERROR')
        epochs_list.append(epochs)

        # Memory clean karna har 50 files ke baad
        if idx % 50 == 0:
            gc.collect()
            print(f"Processed {idx}/{len(target_files)} files...")

    except Exception as e:
        continue

if len(epochs_list) > 0:
    print("\nFiles process ho gayin. Ab inko ek matrix me jorr rahe hain...")
    final_epochs = mne.concatenate_epochs(epochs_list, verbose='ERROR')
    X = final_epochs.get_data(copy=False)
    y = final_epochs.events[:, -1]

    # RAM bachaen
    del epochs_list
    del final_epochs
    gc.collect()

    print(f"\n✅ Data Ready! Massive X Shape: {X.shape}")

    print("\nStep 2: Data Split aur Specialized EEG-CNN banana...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    model = models.Sequential([
        layers.Conv2D(8, (1, 64), padding='same', input_shape=(64, X_train.shape[2], 1), use_bias=False),
        layers.BatchNormalization(),

        layers.DepthwiseConv2D((64, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.5),

        layers.SeparableConv2D(16, (1, 16), use_bias=False, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(2, activation='softmax', kernel_constraint=constraints.max_norm(0.25))
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\nStep 3: Deep Training Shuru...")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("\n✅ --- FULL DATASET TRAINING COMPLETE --- ✅")

    # 📊 Final Result Graphs
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
    plt.title('Final EEG-CNN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Testing Loss')
    plt.title('Final EEG-CNN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

else:
    print("Error: Data load nahi ho paya.")