import cv2
import numpy as np
# Define the eye detector and load the pre-trained Haar cascade classifier
eye_cascade = cv2.CascadeClassifier('../Data/haarcascade_eye.xml')
# Create a window to display the video stream
cv2.namedWindow('Eye Tracking')
# Capture a video stream from the default camera
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect eyes in the grayscale image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # Loop over each detected eye
    for (ex,ey,ew,eh) in eyes:
        # Extract the region of interest (ROI) around the eye
        roi_gray = gray[ey:ey+eh, ex:ex+ew]
        roi_color = frame[ey:ey+eh, ex:ex+ew]
        # Apply a Gaussian blur to reduce noise
        roi_gray = cv2.GaussianBlur(roi_gray, (7, 7), 0)
        # Apply adaptive thresholding to emphasize the iris
        thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort the contours by area in descending order
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        # Loop over each contour and find the center of the iris
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            iris_x = x + w / 2
            iris_y = y + h / 2
            # Draw a circle around the iris center
            cv2.circle(roi_color, (int(iris_x), int(iris_y)), 2, (0, 255, 0), 2)
    # Display the video stream with the iris tracking overlay
    cv2.imshow('Eye Tracking', frame)
    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

## TODO
#Better Camera
#Gaze implementation
#Fine-tune the parameters of the face and eye detectors to better match the characteristics of your user population. For example, if you're developing an eye-tracking system for children, you may need to adjust the detector thresholds to account for smaller and more variable facial features.
#ML to train on large dataset and then: This can help improve the robustness of the system by accounting for individual differences in eye morphology and gaze behavior.
#More users

