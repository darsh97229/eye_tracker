import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import tkinter as tk
from PIL import Image, ImageTk

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye aspect ratio (EAR) threshold and number of consecutive frames to detect
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 15

# Initialize variables to track eye movement
left_ear = right_ear = ear = 0
left_counter = right_counter = 0
blinked = False

# Initialize tkinter GUI
root = tk.Tk()
root.title("Eye Tracking")

# Create canvas to display webcam feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Define function to process each frame of the video feed
def process_frame():
    global left_ear, right_ear, ear, left_counter, right_counter, blinked

    # Read in frame from video feed and convert to grayscale
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop over all detected faces
    for face in faces:
        # Get facial landmarks for the face
        landmarks = predictor(gray, face)

        # Extract the left and right eye landmarks
        left_eye = []
        right_eye = []
        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        # Calculate eye aspect ratio (EAR) for each eye
        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        ear = (left_ear + right_ear) / 2

        # Draw boxes around the eyes
        for eye in [left_eye, right_eye]:
            x, y, w, h = cv2.boundingRect(np.array(eye))
            canvas.create_rectangle(x, y, x + w, y + h, outline='green')

        # for eye in [left_eye, right_eye]:
        #     # Find the coordinates of the top-left and bottom-right corners of the eye rectangle
        #     x1, y1 = np.min(eye, axis=0)
        #     x2, y2 = np.max(eye, axis=0)
        #
        #     # Draw a rectangle around the eye
        #     canvas.create_line(x1, y1, x2, y1, fill='red', width=2)
        #     canvas.create_line(x1, y1, x1, y2, fill='red', width=2)
        #     canvas.create_line(x2, y1, x2, y2, fill='red', width=2)
        #     canvas.create_line(x1, y2, x2, y2, fill='red', width=2)

        # Check if the person blinked
        if ear < EAR_THRESHOLD:
            if not blinked:
                blinked = True
                left_counter = right_counter = 0
                print("Blinked")
        else:
            blinked = False

        # Check if the person is looking left or right
        if left_ear < right_ear:
            left_counter += 1
            right_counter = 0
            if left_counter >= CONSEC_FRAMES:
                print("Looking left")
                left_counter = 0
        elif right_ear < left_ear:
            right_counter += 1
            left_counter = 0
            if right_counter >= CONSEC_FRAMES:
                print("Looking right")
                right_counter = 0
    # # Display the processed frame on the canvas
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    # canvas.create

def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def update_video():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.photo = photo
    process_frame()
    root.after(1, update_video)


update_video()
root.after(0, process_frame)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
