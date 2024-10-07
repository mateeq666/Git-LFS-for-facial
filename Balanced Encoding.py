import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# Set paths
dataset_dir = r"C:\Users\mohamed.ateek\Downloads\Jupyter\Face Recog Original\Phase 2"
train_dir = os.path.join(dataset_dir, 'training')
test_dir = os.path.join(dataset_dir, 'testing')

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

def get_label_from_filename(filename):
    """Determine label from filename ('real' or 'fake')."""
    if 'real' in filename.lower():
        return 'real'
    elif 'fake' in filename.lower():
        return 'spoof'
    else:
        raise ValueError(f"Cannot determine label from filename: {filename}")

def process_and_encode(directory):
    images = []
    labels = []

    # Loop through color and depth folders
    for modality in ['color', 'depth']:
        modality_dir = os.path.join(directory, modality)
        for filename in os.listdir(modality_dir):
            img_path = os.path.join(modality_dir, filename)

            # Load the image
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img)
            img /= 255.0  # Normalize the image

            # Determine label from filename
            label = get_label_from_filename(filename)

            # Append the image and the corresponding label
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Encode labels (real -> 0, spoof -> 1)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels

# Process and encode train and test data
x_train, y_train = process_and_encode(train_dir)
x_test, y_test = process_and_encode(test_dir)

# Save the processed data for future use
np.save(os.path.join(dataset_dir, 'x_train.npy'), x_train)
np.save(os.path.join(dataset_dir, 'y_train.npy'), y_train)
np.save(os.path.join(dataset_dir, 'x_test.npy'), x_test)
np.save(os.path.join(dataset_dir, 'y_test.npy'), y_test)

print("Processing and encoding complete. Data saved as .npy files.")
