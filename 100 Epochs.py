import cv2
import dlib
import numpy as np
import tensorflow as tf

# Load dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()

# Load the shape predictor
predictor_path = r"C:\Users\mohamed.ateek\Downloads\Jupyter\Face Recog Original\shape_predictor_68_face_landmarks.dat"  # Download this file from dlib's model repository
sp = dlib.shape_predictor(predictor_path)

# Load the face recognition model
face_recognition_model_path = r"C:\Users\mohamed.ateek\Downloads\Jupyter\Face Recog Original\dlib_face_recognition_resnet_model_v1.dat"  # Download this file from dlib's model repository
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load the spoof detection model
spoof_model = tf.keras.models.load_model(r"C:\Users\mohamed.ateek\Downloads\Jupyter\best_model.keras")

# Load the face encodings
encodings = np.load(r"C:\Users\mohamed.ateek\Downloads\Jupyter\Face Recog Original\encodings.npy", allow_pickle = True)

def get_face_encodings(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    encodings = []
    for face in faces:
        shape = sp(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, shape)
        encodings.append(np.array(face_descriptor))
    return faces, encodings

def detect_spoof(frame):
    # Preprocess frame for spoof detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (128, 128))
    expanded_frame = np.expand_dims(resized_frame, axis=0)
    
    # Make prediction
    prediction = spoof_model.predict(expanded_frame)
    return prediction[0][0] > 0.7  # Assuming binary classification (spoof or real)

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces, encodings = get_face_encodings(frame)
    
    for (i, face) in enumerate(faces):
        # Draw rectangle around the face
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Check if the face is spoofed
        if detect_spoof(frame[y:y+h, x:x+w]):
            cv2.putText(frame, "Spoofed Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Real Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Face Recognition and Spoof Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
