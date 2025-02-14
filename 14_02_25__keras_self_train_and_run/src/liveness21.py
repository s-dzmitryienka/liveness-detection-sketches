import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Check if the model file exists
model_path = 'liveness_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please provide the correct path or train/download a model.")

# Load the pre-trained liveness detection model
model = load_model(model_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame for the model
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model input size
    normalized_frame = resized_frame.astype('float') / 255.0  # Normalize pixel values
    input_frame = img_to_array(normalized_frame)
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # Perform liveness detection
    predictions = model.predict(input_frame)
    liveness_score = predictions[0][0]  # Assuming the model outputs a single score

    # Display the liveness score on the frame
    cv2.putText(frame, f"Liveness: {liveness_score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # If liveness score is above a threshold, consider it as live
    if liveness_score > 0.7:
        cv2.putText(frame, "Live", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Spoof", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Liveness Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()