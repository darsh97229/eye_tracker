import cv2
# Load the face and eye cascades
face_cascade = cv2.CascadeClassifier("../Data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("../Data/haarcascade_eye.xml")
# Initialize the webcam
cap = cv2.VideoCapture(0)
# Loop over frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Loop over the faces and detect eyes within each face region
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Extract the region of interest (ROI) within the face rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes within the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Loop over the eyes and draw rectangles around them
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # Calculate the center of the eye
            eye_center = (x + ex + ew//2, y + ey + eh//2)
            # Draw a circle at the center of the eye
            cv2.circle(frame, eye_center, 2, (0, 0, 255), 2)
            # Print the coordinates of the eye center to the console
            print(eye_center)
    # Display the annotated frame
    cv2.imshow('frame',frame)
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()