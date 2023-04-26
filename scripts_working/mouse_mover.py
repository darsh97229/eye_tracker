import cv2
import numpy as np
import pyautogui
import win32api

# Set up the eye tracker
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)
detector = cv2.CascadeClassifier('../Data/haarcascade_eye.xml')

# Set up the screen size for mapping
screen_width, screen_height = pyautogui.size()
cv2.resizeWindow('frame', screen_width, screen_height)
print(screen_width // 2)
# Define the calibration points and window size
calibration_points = [(100, 100), (screen_width // 20, 100), (screen_width - 100, 100),
                      (100, screen_height // 20), (screen_width // 20, screen_height // 20), (screen_width - 100, screen_height // 20),
                      (100, screen_height - 100), (screen_width // 20, screen_height - 100), (screen_width - 100, screen_height - 100)]
CALIBRATION_WINDOW_SIZE = 50

# Initialize calibration variables
calibration_complete = False
calibration_point_index = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale and detect eyes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = detector.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected eye
    for (x, y, w, h) in eyes:
        # Calculate the center of the iris
        cx = x + w // 2
        cy = y + h // 2

        # If calibration is not complete, show the calibration window and wait for the user to click on the current point
        if not calibration_complete and calibration_point_index < len(calibration_points):
            # Draw the calibration window
            cv2.rectangle(frame,
                          (calibration_points[calibration_point_index][0] - CALIBRATION_WINDOW_SIZE // 2,
                           calibration_points[calibration_point_index][1] - CALIBRATION_WINDOW_SIZE // 2),
                          (calibration_points[calibration_point_index][0] + CALIBRATION_WINDOW_SIZE // 2,
                           calibration_points[calibration_point_index][1] + CALIBRATION_WINDOW_SIZE // 2),
                          (0, 0, 255), 2)

            # Check if the user has clicked on the calibration point
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Move to the next calibration point
                calibration_point_index += 1
                print(f"Calibrating point {calibration_point_index} of {len(calibration_points)}")

                # Wait for a short delay to give the user time to look at the next point
                cv2.waitKey(1000)

                # If all calibration points have been visited, calibration is complete
                if calibration_point_index == len(calibration_points):
                    calibration_complete = True
                    print("Calibration complete!")

        elif calibration_complete:
            # If calibration is complete, map the iris position to the mouse cursor position
            cursor_x = int(cx / w * screen_width)
            cursor_y = int(cy / h * screen_height)
            win32api.SetCursorPos((cursor_x, cursor_y))

    # Show the frame
    cv2.imshow('frame', frame)

    # Quit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
