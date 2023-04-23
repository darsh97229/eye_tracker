import cv2
import numpy as np
# Load the face and eye detectors
face_cascade = cv2.CascadeClassifier('../Data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../Data/haarcascade_eye.xml')
# Start the video capture
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Process each face in the image
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Crop the region of the face containing the eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes within the region of the face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Process each eye in the image
        for (ex,ey,ew,eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # Calculate the center of the eye
            eye_center_x = x + ex + ew / 2
            eye_center_y = y + ey + eh / 2
            # Calculate the distance between the centers of the eyes
            eye_distance = np.sqrt((ew**2 + eh**2))
            # Use the distance between the eyes to estimate the distance to the screen
            screen_distance = 500 / eye_distance  # assuming a constant screen size of 500 pixels
            # Print the estimated distance to the screen
            print('Estimated distance to screen: {:.2f} cm'.format(screen_distance))
    # Display the processed video frame
    cv2.imshow('frame',frame)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()