
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion detection model
loaded_model = load_model('emotiondetectionmodel.h5')

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
cap = cv2.VideoCapture(0)

def predict_emotion(frame):
    if frame is None:
        print("Error: Frame is None")
        return None

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # Emotion detection
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))

        # Predict emotion
        predicted_emotion = np.argmax(loaded_model.predict(np.expand_dims(face_roi, axis=0)))
        emotion_label = emotions[predicted_emotion]

        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# Verify model architecture and input shape
print(loaded_model.summary())
input_shape = loaded_model.input_shape
print("Model Input Shape:", input_shape)

# Real-time emotion detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not read from webcam")
        break

    # Debugging: Print out input data shape and type before prediction
    print("Input Data Shape:", frame.shape)
    print("Input Data Type:", frame.dtype)

    detected_frame = predict_emotion(frame)
    if detected_frame is not None:
        cv2.imshow('Emotion Detection', detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
