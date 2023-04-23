import cv2
import numpy as np
# Initialize HOG+SVM detector for eyes
eye_detector = cv2.HOGDescriptor()
eye_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Initialize Kalman filter for eye tracking
state_size = 4
measure_size = 2
kalman = cv2.KalmanFilter(state_size, measure_size)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1e-3, 0, 0, 0], [0, 1e-3, 0, 0], [0, 0, 1e-3, 0], [0, 0, 0, 1e-3]], np.float32) * 0.1
kalman.measurementNoiseCov = np.array([[1e-1, 0], [0, 1e-1]], np.float32) * 0.1
# Initialize video capture
cap = cv2.VideoCapture(0)
# Main loop
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect eyes using HOG+SVM detector
    eyes = eye_detector.detectMultiScale(gray)
    # Initialize Kalman filter state with first detected eye
    if len(eyes) > 0:
        print(eyes)
        x, y, w, h = eyes[0]
        kalman.statePost = np.array([[x+w/2], [y+h/2], [0], [0]], np.float32)
    # Predict eye position using Kalman filter
    kalman.predict()
    # Draw predicted eye position on frame
    predicted_state = kalman.statePre
    x, y = predicted_state[0][0], predicted_state[1][0]
    cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)
    # Update Kalman filter with detected eye position
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        kalman.correct(np.array([[x+w/2], [y+h/2]], np.float32))
    # Display frame
    cv2.imshow('Eye Tracker', frame)
    # Wait for user input
    if cv2.waitKey(1) == ord('q'):
        break
# Clean up resources
cap.release()
cv2.destroyAllWindows()
