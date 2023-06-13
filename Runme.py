from keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('Model-45.model')

# Load the Haar cascade classifier for face detection
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the video capture device
source = cv2.VideoCapture(0)

# Define dictionaries for labels and colors
labels_dict = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 255, 255), 5: (255, 0, 255), 6: (255, 255, 255)}

while True:
    # Read the video frame
    ret, img = source.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)  

    for (x, y, w, h) in faces:
        # Extract the face region from the grayscale image
        face_img = gray[y:y+w, x:x+w]

        # Resize the face image to match the input size of the model
        resized = cv2.resize(face_img, (48, 48))

        # Convert the resized image to RGB color space
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        # Normalize the RGB image
        normalized = rgb / 255.0

        # Reshape the normalized image to match the input shape of the model
        reshaped = np.reshape(normalized, (1, 48, 48, 3))

        # Make predictions using the model
        result = model.predict(reshaped)

        # Get the predicted label
        label = np.argmax(result, axis=1)[0]

        # Draw bounding box and label on the original image
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display the image with detected emotions
    cv2.imshow('Emotion Detection', img)

    # Check for the 'Esc' key to exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

# Close all windows and release the video capture device
cv2.destroyAllWindows()
source.release()
