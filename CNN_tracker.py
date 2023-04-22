import cv2
import numpy as np
import tensorflow as tf
# Load the pre-trained eye detection model
model = tf.keras.models.load_model('eye_detection_model.h5')
# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to match the input size of the CNN
    resized = cv2.resize(gray, (64, 64))
    # Normalize the pixel values to be between 0 and 1
    normalized = resized / 255.0
    # Add an extra dimension to the image array to match the input shape of the CNN
    input_data = np.expand_dims(normalized, axis=0)
    # Use the eye detection model to make a prediction on the input image
    prediction = model.predict(input_data)
    # Extract the predicted eye location from the output of the model
    eye_x, eye_y = prediction[0]
    # Draw a circle around the predicted eye location on the original frame
    cv2.circle(frame, (int(eye_x * frame.shape[1]), int(eye_y * frame.shape[0])), 10, (0, 0, 255), -1)
    # Display the resulting frame
    cv2.imshow('Eye Tracker', frame)
    # Wait for a key press and check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()