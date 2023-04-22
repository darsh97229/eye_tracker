import cv2
import dlib
# Initialize face detector and eye detector
face_detector = dlib.get_frontal_face_detector()
eye_detector = dlib.get_frontal_face_detector()
# Initialize video capture device
cap = cv2.VideoCapture(0)
# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Convert the frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_detector(gray, 0)
    # Loop over detected faces
    for face in faces:
        # Detect eyes within the region of the face
        eyes = eye_detector(gray, face)
        # Loop over detected eyes
        for eye in eyes:
            # Draw a rectangle around each eye
            x, y, w, h = eye.left(), eye.top(), eye.width(), eye.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Calculate the center of the eye
            cx, cy = x + w/2, y + h/2
            # Use the center of the eye to determine where the user is looking
            # (you can use this information for further processing)
            if cx < frame.shape[1]/3:
                direction = "left"
            elif cx > frame.shape[1]*2/3:
                direction = "right"
            else:
                direction = "center"
            # Draw a circle at the center of the eye
            cv2.circle(frame, (int(cx), int(cy)), 2, (0, 0, 255), -1)
    # Display the resulting frame
    cv2.imshow("Eye Tracking", frame)
    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
