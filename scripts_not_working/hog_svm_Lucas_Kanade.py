import cv2
# Initialize HOG+SVM detectors for face and eye detection
face_detector = cv2.CascadeClassifier('../Data/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('../Data/haarcascade_eye.xml')
# Initialize Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Initialize video capture from default camera
cap = cv2.VideoCapture(0)
# Initialize previous points for Lucas-Kanade optical flow
prev_points = None
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    # Convert frame to grayscale for HOG+SVM detectors
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use HOG+SVM detectors to detect face and eyes
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_detector.detectMultiScale(roi_gray)
        # Use Lucas-Kanade optical flow to track eye movement
        if len(eyes) == 2:
            # Extract eye regions from face ROI
            eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
            eye2_x, eye2_y, eye2_w, eye2_h = eyes[1]
            eye1_roi = roi_gray[eye1_y:eye1_y + eye1_h, eye1_x:eye1_x + eye1_w]
            eye2_roi = roi_gray[eye2_y:eye2_y + eye2_h, eye2_x:eye2_x + eye2_w]
            # Use the first frame as the previous frame for optical flow
            if prev_points is None:
                prev_points = cv2.goodFeaturesToTrack(eye1_roi, 100, 0.3, 7, blockSize=7)
            # Calculate optical flow for first eye
            curr_points, status, err = cv2.calcOpticalFlowPyrLK(eye1_roi, gray[y + eye1_y:y + eye1_y + eye1_h,
                                                                             x + eye1_x:x + eye1_x + eye1_w],
                                                                prev_points, None, **lk_params)
            # Calculate the average movement of the optical flow points
            avg_movement = curr_points.mean(axis=0) - prev_points.mean(axis=0)

            # Draw a line showing the direction of eye movement
            cv2.line(frame, (x + eye1_x + int(eye1_w / 2), y + eye1_y + int(eye1_h / 2)),
                     (x + eye1_x + int(eye1_w / 2) + int(avg_movement[0][0]), y + eye1_y + int(eye1_h / 2) + int(avg_movement[0][1])),
                     (0, 255, 0), 2)
            # Update previous points for next frame
            prev_points = curr_points
    # Show the frame with eye movement tracking
    cv2.imshow('Eye Movement Tracker', frame)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()