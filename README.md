# Emotion Detection Using OpenCV

This project implements an emotion detection system using deep learning techniques. It can detect emotions like anger, disgust, fear, happiness, neutrality, sadness, and surprise from facial images. The project consists of two main scripts: `train.py` and `runme.py`. 

### Prerequisites
Download all the zip files containing 48x48 pixel face images representing different emotions and place them in a single directory called "Data". Make sure to keep the directory structure consistent with the provided scripts.

### Training (train.py)
The `train.py` script is used to train the emotion detection model. It follows these steps:

1. Loads the facial images dataset from the specified directory.
2. Preprocesses the images, including grayscale conversion and resizing.
3. Splits the dataset into training and testing sets.
4. Constructs a convolutional neural network (CNN) model architecture.
5. Compiles the model with an appropriate loss function and optimizer.
6. Trains the model on the training set and saves the best model weights.
7. Evaluates the model on the testing set and displays accuracy and loss metrics.

To run the `train.py` script, ensure you have the necessary dependencies installed (specified in the `requirements.txt` file). Execute the following command:

```
python train.py
```

### Real-Time Emotion Detection (runme.py)
The `runme.py` script is used to perform real-time emotion detection on a video stream. It utilizes the trained model to detect emotions from the live webcam feed. The script follows these steps:

1. Loads the pre-trained emotion detection model.
2. Uses the OpenCV library to capture video frames from the webcam.
3. Detects faces in the frames using the Haar cascade classifier.
4. Extracts the face region and preprocesses it for input to the model.
5. Makes predictions using the loaded model and assigns corresponding emotion labels.
6. Draws bounding boxes and labels on the frames.
7. Displays the processed frames with detected emotions in real-time.

To run the `runme.py` script, ensure you have the necessary dependencies installed (specified in the `requirements.txt` file). Execute the following command:

```
python runme.py
```

Please make sure to download the necessary image dataset and place it in the correct directory before running the scripts. The `train.py` script will train the model and save the best model as `Model-45.model` with an accuracy of approximately 65%. The `runme.py` script will utilize the trained model to detect emotions from the webcam feed in real-time.
