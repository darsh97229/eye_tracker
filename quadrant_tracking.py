import cv2
# Initialize the face and eye detectors
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Define the screen quadrants
width, height = 640, 480
quadrants = {
    'top_left': (0, 0, int(width/2), int(height/2)),
    'top_right': (int(width/2), 0, width, int(height/2)),
    'bottom_left': (0, int(height/2), int(width/2), height),
    'bottom_right': (int(width/2), int(height/2), width, height)
}
# Initialize the video capture
cap = cv2.VideoCapture(0)
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Loop over each detected face
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        # Extract the region of the face containing the eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes in the eye region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Loop over each detected eye
        for (ex,ey,ew,eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            # Calculate the center of the eye
            eye_center = (x+ex+ew//2, y+ey+eh//2)
            # Determine which quadrant the eye is in
            for quadrant, (x1, y1, x2, y2) in quadrants.items():
                if x1 <= eye_center[0] <= x2 and y1 <= eye_center[1] <= y2:
                    print('Eye is in', quadrant)
                    break
    # Display the frame with eye tracking overlays
    cv2.imshow('Eye Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()