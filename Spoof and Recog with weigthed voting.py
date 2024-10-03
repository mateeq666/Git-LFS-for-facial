import cv2
import dlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import time

# Load dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor_path = r"C:\Users\mohamed.ateek\Downloads\Jupyter\Face Recog Original\shape_predictor_68_face_landmarks.dat"
sp = dlib.shape_predictor(predictor_path)
face_recognition_model_path = r"C:\Users\mohamed.ateek\Downloads\Jupyter\Face Recog Original\dlib_face_recognition_resnet_model_v1.dat"
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load the spoof detection model
spoof_model = tf.keras.models.load_model(r"C:\Users\mohamed.ateek\Downloads\Jupyter\best_model.keras")

# Load the feature vectors from CSV
feature_vector_csv_path = r"C:\Users\mohamed.ateek\Downloads\embeddings.csv"
df = pd.read_csv(feature_vector_csv_path)
names = df['Person_Name'].values
features = df.drop(columns=['Person_Name']).values

def detect_spoof(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (128, 128))
    expanded_frame = np.expand_dims(resized_frame, axis=0)
    prediction = spoof_model.predict(expanded_frame)
    return prediction[0][0] > 0.7

def get_face_encodings(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    encodings = []
    for face in faces:
        shape = sp(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, shape)
        encodings.append(np.array(face_descriptor))
    return faces, encodings

def weighted_voting(encoding):
    similarities = cosine_similarity([encoding], features)[0]
    euclideans = np.array([euclidean(encoding, feature) for feature in features])

    # Normalize distances
    norm_similarities = similarities / np.sum(similarities)
    norm_euclideans = 1 / (1 + euclideans)  # Higher similarity with smaller distances
    norm_euclideans = norm_euclideans / np.sum(norm_euclideans)

    # Weighted voting (50% cosine, 50% euclidean)
    final_scores = 0.5 * norm_similarities + 0.5 * norm_euclideans
    index = np.argmax(final_scores)

    if final_scores[index] > 0.8:  # Adjust threshold as needed
        return names[index]
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, encodings = get_face_encodings(frame)
    spoof_detected = False

    for (i, face) in enumerate(faces):
        face_region = frame[face.top():face.bottom(), face.left():face.right()]

        if detect_spoof(face_region):
            cv2.putText(frame, "Remove Spoofed Image", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            spoof_detected = True
            break

    if not spoof_detected:
        for (i, face) in enumerate(faces):
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(encodings) > i:
                name = weighted_voting(encodings[i])
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if time.time() - start_time >= 5:
                    print(f"Attendance log: {name}")
                    start_time = time.time()  # Reset the timer every 5 seconds
            else:
                cv2.putText(frame, "No Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

