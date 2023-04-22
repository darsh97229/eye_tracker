import cv2
import numpy as np
# initialize video capture from default camera
cap = cv2.VideoCapture(0)
# initialize face and eye detectors
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# initialize parameters for optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# initialize variables for tracking optical flow
old_gray = None
old_pts = None
while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # loop over all detected faces
    for (x, y, w, h) in faces:
        # extract the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # loop over all detected eyes
        for (ex, ey, ew, eh) in eyes:
            # extract the eye region
            eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_color = roi_color[ey:ey + eh, ex:ex + ew]
            # initialize optical flow if this is the first time we're seeing this eye
            if old_gray is None:
                old_gray = eye_gray
                old_pts = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, mask=None, **lk_params)
            # calculate optical flow between previous and current eye frames
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray, eye_gray, old_pts, None, **lk_params)
            # filter out points with bad status
            good_new = new_pts[status == 1]
            good_old = old_pts[status == 1]
            # compute the average movement of the eye in pixels
            dx = np.mean(good_new[:, 0, 0] - good_old[:, 0, 0])
            dy = np.mean(good_new[:, 0, 1] - good_old[:, 0, 1])
            # update the old points with the new points for the next frame
            old_gray = eye_gray.copy()
            old_pts = good_new.reshape(-1, 1, 2)
            # draw a line to visualize the direction of eye movement
            cv2.line(eye_color, (int(ew / 2), int(eh / 2)), (int(ew / 2 + dx), int(eh / 2 + dy)), (0, 255, 0), 2)
    # display the resulting frame
    cv2.imshow('frame', frame)
    # exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()