import cv2
import easyocr
import pyttsx3
import numpy as np
import matplotlib.pyplot as plt

# Initialize OCR and Text-to-Speech
reader = easyocr.Reader(['en'])
engine = pyttsx3.init()

def read_text_from_image():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not access camera.")
            break

        # Convert frame to RGB for OCR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform OCR to extract text
        results = reader.readtext(frame_rgb, detail=0)
        text = " ".join(results)

        # Display detected text on the frame
        for i, line in enumerate(results):
            cv2.putText(frame, line, (50, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame using Matplotlib instead of OpenCV
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.title("Text Detection - Press Ctrl+C to Stop")
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()

        # Speak the detected text
        if text.strip():
            print("\n📖 Detected Text:", text)
            engine.say(text)
            engine.runAndWait()

    cap.release()
    plt.close()

if __name__ == "__main__":
    read_text_from_image()
