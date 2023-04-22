import cv2
import dlib
import math
# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Load arrow image
arrow_image = cv2.imread("arrow.png", cv2.IMREAD_UNCHANGED)
# Define arrow scaling factor and length in pixels
arrow_scale = 0.3
arrow_length = 100
# Define arrow color and thickness
arrow_color = (255, 0, 0)
arrow_thickness = 2
# Define video capture device
cap = cv2.VideoCapture(0)
while True:
    # Capture frame from video stream
    ret, frame = cap.read()
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in grayscale frame
    faces = detector(gray)
    # For each detected face, find facial landmarks and estimate gaze direction
    for face in faces:
        # Find facial landmarks
        landmarks = predictor(gray, face)
        # Extract eye landmarks
        left_eye = [(landmarks.part(36).x, landmarks.part(36).y),
                    (landmarks.part(37).x, landmarks.part(37).y),
                    (landmarks.part(38).x, landmarks.part(38).y),
                    (landmarks.part(39).x, landmarks.part(39).y),
                    (landmarks.part(40).x, landmarks.part(40).y),
                    (landmarks.part(41).x, landmarks.part(41).y)]
        right_eye = [(landmarks.part(42).x, landmarks.part(42).y),
                     (landmarks.part(43).x, landmarks.part(43).y),
                     (landmarks.part(44).x, landmarks.part(44).y),
                     (landmarks.part(45).x, landmarks.part(45).y),
                     (landmarks.part(46).x, landmarks.part(46).y),
                     (landmarks.part(47).x, landmarks.part(47).y)]
        # Compute center of left and right eyes
        left_eye_center = (int(sum([pt[0] for pt in left_eye]) / 6),
                           int(sum([pt[1] for pt in left_eye]) / 6))
        right_eye_center = (int(sum([pt[0] for pt in right_eye]) / 6),
                            int(sum([pt[1] for pt in right_eye]) / 6))
        # Compute angle between eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = math.atan2(dy, dx) * 180 / math.pi
        # Compute endpoint of arrow
        arrow_endpoint = (int(right_eye_center[0] + arrow_length * math.cos(angle)),
                          int(right_eye_center[1] + arrow_length * math.sin(angle)))
        # Resize arrow
        arrow_resized = cv2.resize(arrow_image, None, fx=arrow_scale, fy=arrow_scale)
        # Rotate arrow
        rows, cols, _ = arrow_resized.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
        arrow_rotated = cv2.warpAffine(arrow_resized, M, (cols, rows))
        # Draw arrow on frame
        cv2.arrowedLine(frame, right_eye_center, arrow_endpoint, arrow_color, arrow_thickness)
        overlay = arrow_rotated[:, :, :3]
        mask = arrow_rotated[:, :, 3]
        mask_inv = cv2.bitwise_not(mask)
        roi = frame[right_eye_center[1] - rows // 2:right_eye_center[1] + rows // 2,
                    right_eye_center[0] - cols // 2:right_eye_center[0] + cols // 2]
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(overlay, overlay, mask=mask)
        dst = cv2.add(bg, fg)
        frame[right_eye_center[1] - rows // 2:right_eye_center[1] + rows // 2, right_eye_center[0] - cols // 2:right_eye_center[0] + cols // 2] = dst
    # Display frame
    cv2.imshow("Frame", frame)
    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
