import cv2
import os
import numpy as np
import pyttsx3
import threading
import time
import streamlit as st
from ultralytics import YOLO  # YOLOv8

# Constants
DATASET_PATH = "datasets"
MODEL_PATH = "face_model.xml"

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")

# Voice Assistant Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Cooldown Storage for Speech Output
last_spoken = {}
cooldown_time = 5  
previous_detections = set()  

def speak_text(text):
    """Speaks text but prevents repeating the same text within cooldown time."""
    current_time = time.time()
    if text in last_spoken and (current_time - last_spoken[text] < cooldown_time):
        return  
    last_spoken[text] = current_time  
    thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()))
    thread.start()

# ✅ Function to Create Face Dataset
def create_dataset(person_name):
    """Captures images and saves them for training the face recognition model."""
    person_path = os.path.join(DATASET_PATH, person_name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    st.write(f"Capturing images for {person_name}. Press 'ESC' to stop.")

    count = 0
    while count < 50:  
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (130, 100))
            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face_resized)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/50", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Dataset Collection", frame)
        if cv2.waitKey(10) == 27 or count >= 50:  
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success(f"Dataset collected for {person_name}. Training the model...")

    # ✅ Automatically train the model after dataset collection
    train_face_model()

# ✅ Function to Train Face Recognition Model
def train_face_model():
    """Trains the Face Recognition Model on collected images and saves it."""
    
    # Updated: Using cv2.face_LBPHFaceRecognizer for compatibility
    face_recognizer = cv2.face_LBPHFaceRecognizer.create()  # Ensure you're using the correct method.
    faces, labels = [], []
    label_dict = {}  
    label_id = 0  

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in label_dict:
            label_dict[person_name] = label_id
            label_id += 1

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_dict[person_name])

    if not faces:
        st.error("No faces found for training. Add more data!")
        return

    # Train and save the model
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save(MODEL_PATH)
    
    st.success("Model trained successfully! You can now start detection.")

# ✅ Function for Face Recognition and Object Detection using YOLOv8
def recognize_faces_and_detect_objects():
    global previous_detections  

    if not os.path.exists(MODEL_PATH):
        st.error("No trained face model found! Train the model first.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Updated: Using cv2.face_LBPHFaceRecognizer.create()
    model = cv2.face_LBPHFaceRecognizer.create()

    model.read(MODEL_PATH)

    label_dict = {id: name for id, name in enumerate(os.listdir(DATASET_PATH))}

    cap = cv2.VideoCapture(0)
    st.write("Face & Object Detection started. Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_names = set()  

        # Face Recognition
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (130, 100))
            label, confidence = model.predict(face_resized)

            if confidence < 100:
                name = label_dict.get(label, "Unknown")
                detected_names.add(name)
                cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y-10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                detected_names.add("Unknown person")
                cv2.putText(frame, "Unknown", (x, y-10), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # ✅ Object Detection using YOLOv8
        results = yolo_model(frame)
        detected_objects = set()  

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf[0].item()
            cls = int(result.cls[0].item())
            label = yolo_model.names[cls]

            if conf > 0.5:  
                detected_objects.add(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ Speak Only New Detections
        new_detections = detected_names.union(detected_objects) - previous_detections
        if new_detections:  
            speak_text(", ".join(new_detections) + " detected")

        previous_detections = detected_names.union(detected_objects)  

        cv2.imshow("Face & Object Detection", frame)
        if cv2.waitKey(10) == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Streamlit UI
st.title("Face & Object Detection (YOLOv8)")

# Dataset Collection
person_name = st.text_input("Enter name for dataset collection:")
if st.button("Create Dataset"):
    if person_name:
        create_dataset(person_name)
    else:
        st.error("Please enter a valid name.")

# Start Detection
if st.button("Start Detection"):
    recognize_faces_and_detect_objects()
