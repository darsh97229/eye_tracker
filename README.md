# eye_tracker
Eye Tracker: Computer Vision Based
This code initializes the camera and uses the Haar Cascade classifier to detect faces and eyes in each frame. For each eye detected, a rectangle is drawn around it.

http://dlib.net/files/

CNN
1. Download and extract the EYEDB dataset from
http://mrl.cs.vsb.cz/eyedataset
2. Preprocess the dataset by resizing the images and normalizing the pixel values.
3. Split the dataset into training and validation sets.
4. Define a CNN architecture for eye detection.
5. Train the model on the training set using an appropriate loss function and optimizer.
6. Evaluate the model on the validation set.
