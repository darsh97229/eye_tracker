import cv2
import dlib
import numpy as np
import pyautogui

# Define the eye detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../Data/shape_predictor_68_face_landmarks.dat')

# Define the size of the eye region
eye_size = (64, 64)

# Define the amount of eye movement
movement_scale = 15

# Define the minimum and maximum eye aspect ratios
min_aspect_ratio = 0.2
max_aspect_ratio = 0.7

# Define the maximum amount of head movement
max_head_movement = 30

# Define the video capture device
cap = cv2.VideoCapture(0)

# Get the screen dimensions
screen_width, screen_height = pyautogui.size()

# Get the initial mouse position
mouse_x, mouse_y = pyautogui.position()

# Loop over frames from the video capture device
while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop over the detected faces
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)

        # Get the left and right eye regions
        left_eye = get_eye_region(frame, landmarks, 'left')
        right_eye = get_eye_region(frame, landmarks, 'right')

        # Preprocess the left and right eye regions
        left_eye, _, _ = preprocess_eye(left_eye)
        right_eye, _, _ = preprocess_eye(right_eye)

        # Compute the left and right eye aspect ratios
        left_aspect_ratio = compute_aspect_ratio(left_eye)
        right_aspect_ratio = compute_aspect_ratio(right_eye)

        # Get the average eye aspect ratio
        aspect_ratio = (left_aspect_ratio + right_aspect_ratio) / 2

        # Check if the eye aspect ratio is within the acceptable range
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            # Compute the horizontal and vertical eye centers
            left_center = get_eye_center(landmarks, 'left')
            right_center = get_eye_center(landmarks, 'right')
            eye_center = ((left_center[0] + right_center[0]) / 2, (left_center[1] + right_center[1]) / 2)

            # Compute the amount of eye movement
            dx = (eye_center[0] - screen_width / 2) / (screen_width / 2) * movement_scale
            dy = (eye_center[1] - screen_height / 2) / (screen_height / 2) * movement_scale

            # Move the mouse cursor by the amount of eye movement
            mouse_x += dx
            mouse_y += dy

            # Check if the mouse cursor is within the screen boundaries
            if mouse_x < 0:
                mouse_x = 0
            elif mouse_x > screen_width:
                mouse_x = screen_width
            if mouse_y < 0:
                mouse_y = 0
            elif mouse_y > screen_height:
                mouse_y = screen_height

            # Move the mouse cursor to the new position
            pyautogui.moveTo(mouse_x, mouse_y)

    # Show the video frame
    cv2.imshow('Eye Tracker', frame)

    # Check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()