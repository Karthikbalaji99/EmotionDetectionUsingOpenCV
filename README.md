# EmotionDetectionUsingOpenCV
This project implements an emotion detection system using deep learning techniques. It can detect emotions like anger, disgust, fear, happiness, neutrality, sadness, and surprise from facial images. The project consists of two main scripts: train.py and runme.py. 
Download all the zip files- they contain 48x48 pixels of face extracted image which shows respective emotions - and place all the files in a single directory same as the py scripts and call that folder "Data". Then you can run the train.py and save the best model.
train.py
The train.py script is used to train the emotion detection model. It uses a convolutional neural network (CNN) architecture to learn and classify facial expressions. The script performs the following steps:

Loads the facial images dataset from the specified directory.
Preprocesses the images, including grayscale conversion and resizing.
Splits the dataset into training and testing sets.
Constructs the CNN model architecture.
Compiles the model with appropriate loss function and optimizer.
Trains the model on the training set and saves the best model weights.
Evaluates the model on the testing set and displays the accuracy and loss metrics.
We save the best model.. ours is Model -45.model with an accuracy of around 65%.
To run the train.py script, make sure you have the necessary dependencies installed (specified in the requirements.txt file) and execute the following command:
python train.py

runme.py

The runme.py script is used to run real-time emotion detection on a video stream. It utilizes the trained model to detect emotions from live webcam feed. The script performs the following steps:

Loads the pre-trained emotion detection model.
Uses the OpenCV library to capture video frames from the webcam.
Detects faces in the frames using the Haar cascade classifier.
Extracts the face region and preprocesses it for input to the model.
Makes predictions using the loaded model and assigns corresponding emotion labels.
Draws bounding boxes and emotion labels on the frames.
Displays the processed frames with detected emotions in real-time.
To run the runme.py script, make sure you have the necessary dependencies installed (specified in the requirements.txt file) and execute the following command:
python runme.py
