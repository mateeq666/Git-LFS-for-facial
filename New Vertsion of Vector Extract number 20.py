import os
import dlib
import csv
import numpy as np
import logging
import cv2
from tqdm import tqdm

# Paths
path_images_from_camera = r"C:\Users\mohamed.ateek\Downloads\Docker\images"
shape_predictor_path = r"C:\Users\mohamed.ateek\Downloads\VS Files\face_recognitionenv\shape_predictor_68_face_landmarks.dat"
cnn_face_detector_path = r"C:\Users\mohamed.ateek\Downloads\VS Files\face_recognitionenv\mmod_human_face_detector.dat"
face_reco_model_path = r"C:\Users\mohamed.ateek\Downloads\VS Files\face_recognitionenv\dlib_face_recognition_resnet_model_v1.dat"
features_csv_path = r"C:\Users\mohamed.ateek\Downloads\VS Files\SDD\synthetic_feature_vectors.csv"

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Dlib tools
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)
predictor = dlib.shape_predictor(shape_predictor_path)
face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def augment_image(image):
    """Create synthetic images with random transformations."""
    augmented_images = []

    # Apply transformations: rotation, flip, brightness change, etc.
    flipped = cv2.flip(image, 1)  # Horizontal flip
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotation
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Brightness adjustment

    augmented_images.extend([flipped, rotated, bright])

    return augmented_images

def return_128d_features(image):
    image = preprocess_image(image)
    faces = cnn_face_detector(image, 1)
    if len(faces) == 0:
        return None
    
    face = faces[0].rect
    shape = predictor(image, face)
    face_descriptor = face_reco_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

def return_features_mean_personX(path_face_personX):
    if not os.path.isdir(path_face_personX):
        logging.error(f"Directory not found: {path_face_personX}")
        return np.zeros(128)
    
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    
    if not photos_list:
        logging.warning(f"No images in {path_face_personX}")
        return np.zeros(128)

    for photo in photos_list:
        photo_path = os.path.join(path_face_personX, photo)
        image = cv2.imread(photo_path)
        if image is None:
            logging.warning(f"Unable to read image {photo_path}")
            continue
        
        # Original image feature extraction
        features_128d = return_128d_features(image)
        if features_128d is not None:
            features_list_personX.append(features_128d)

        # Generate synthetic data and extract features
        augmented_images = augment_image(image)
        for aug_img in augmented_images:
            synthetic_features_128d = return_128d_features(aug_img)
            if synthetic_features_128d is not None:
                features_list_personX.append(synthetic_features_128d)

    if features_list_personX:
        features_mean_personX = np.mean(features_list_personX, axis=0)
    else:
        logging.warning(f"No valid features collected for {path_face_personX}")
        features_mean_personX = np.zeros(128)

    return features_mean_personX

def main():
    if not os.path.isdir(path_images_from_camera):
        logging.error(f"Images directory not found: {path_images_from_camera}")
        return

    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    with open(features_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Person_Name"] + [f"Feature_{i}" for i in range(128)])

        for person in tqdm(person_list):
            person_path = os.path.join(path_images_from_camera, person)
            logging.info(f"Processing directory: {person_path}")
            features_mean_personX = return_features_mean_personX(person_path)

            person_name = person.split('_', 2)[-1] if len(person.split('_', 2)) == 3 else person
            row = [person_name] + features_mean_personX.tolist()
            writer.writerow(row)
        
    logging.info(f"Saved all features of faces to: {features_csv_path}")

if __name__ == '__main__':
    main()
