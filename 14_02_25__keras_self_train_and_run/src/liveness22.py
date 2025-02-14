import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained liveness detection model
model = load_model('liveness_model.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, display a message
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess the face region for the liveness detection model
            resized_face = cv2.resize(face_roi, (224, 224))  # Resize to model input size
            normalized_face = resized_face.astype('float') / 255.0  # Normalize pixel values
            input_face = img_to_array(normalized_face)
            input_face = np.expand_dims(input_face, axis=0)  # Add batch dimension

            # Perform liveness detection
            predictions = model.predict(input_face)
            liveness_score = predictions[0][0]  # Assuming the model outputs a single score

            # Display the liveness score on the frame
            cv2.putText(frame, f"Liveness: {liveness_score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # If liveness score is above a threshold, consider it as live
            if liveness_score > 0.5:
                cv2.putText(frame, "Live", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Spoof", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Liveness Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()