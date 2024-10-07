import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the processed data
dataset_dir = r"C:\Users\mohamed.ateek\Downloads\Jupyter\Face Recog Original\Phase 2"
x_train = np.load(os.path.join(dataset_dir, 'x_train.npy'))
y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
x_test = np.load(os.path.join(dataset_dir, 'x_test.npy'))
y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Build the CNN model
def build_cnn_model(input_shape=(128, 128, 3)):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

# Instantiate and compile the model
cnn_model = build_cnn_model()
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
cnn_model.summary()

# Define a callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint('best_model.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

# Train the CNN model with class weights and the checkpoint callback
history = cnn_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[checkpoint]  # Add the checkpoint callback here
)

# Save the final model after the last epoch (if you want to keep it)
cnn_model.save('final_model.keras')

print("Training complete. The best model has been saved as 'best_model.h5', and the final model has been saved as 'final_model.h5'.")
