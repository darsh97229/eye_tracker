import cv2
# Initialize the webcam
cap = cv2.VideoCapture(0)
# Create a face detector
face_cascade = cv2.CascadeClassifier('../Data/haarcascade_frontalface_default.xml')
# Create an eye detector
eye_cascade = cv2.CascadeClassifier('../Data/haarcascade_eye.xml')
# Initialize the mean-shift tracker
tracking_window = None
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # If a face is detected, track the eyes
    if len(faces) > 0:
        # Get the region of interest (ROI) around the face
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # If both eyes are detected, track them using mean-shift
        if len(eyes) == 2:
            # Initialize the tracking window if it hasn't been set yet
            if tracking_window is None:
                ex, ey, ew, eh = eyes[0]
                tracking_window = (ex, ey, ew, eh)
            # Otherwise, update the tracking window based on the mean-shift algorithm
            else:
                # Convert the tracking window to the format required by cv2.meanShift()
                x, y, w, h = tracking_window
                track_window = (x, y, w, h)
                # Calculate the histogram of the current tracking window
                roi_hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                # Apply mean-shift to find the new position of the eyes
                ret, tracking_window = cv2.meanShift(roi_hist, track_window,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
            # Draw a rectangle around each eye
            ex, ey, ew, eh = tracking_window
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # Determine the center of each eye
            eye_centers = [(ex + ew//2, ey + eh//2) for (ex, ey, ew, eh) in eyes]
            # Calculate the distance between the centers of the eyes
            eye_distance = abs(eye_centers[0][0] - eye_centers[1][0])
            # Determine the position of the user's gaze based on the eye distance
            if eye_distance < 80:
                gaze_position = "center"
            elif eye_centers[0][0] < eye_centers[1][0]:
                gaze_position = "left"
            else:
                gaze_position = "right"
            # Display the gaze position on the screen
            cv2.putText(frame, gaze_position, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
