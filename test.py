import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

# switch red and blue color channels
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect all faces in an image
# eye detections https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html
# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# modify these parameters to find faces in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)


## Loading in a trained model
import torch
from models import Net

net = Net()

# load saved model parameters
net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))


# print out the net and prepare it for testing
net.eval()


# Keypoint detection

### Transform each detected face into an input Tensor

# perform the following steps for each detected face:
# 1. Convert the face from RGB to grayscale
# 2. Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
# 3. Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
# 4. Reshape the numpy image into a torch image.

# Detect and display the predicted keypoints
def visualize_output(faces, test_outputs):
    batch_size = len(faces)
    for i, face in enumerate(faces):
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts*50.0+100

        plt.imshow(face, cmap='gray')
        plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')

        plt.axis('off')

    plt.show()

image_copy = np.copy(image)
#Including a padding to extract face as  HAAR classifier's bounding box, crops sections of the face

PADDING = 40
images, keypoints = [], []

# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:

    # Select the region of interest that is the face in the image
    roi = image_copy[y-PADDING:y+h+PADDING, x-PADDING:x+w+PADDING]

    # Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = (roi / 255.).astype(np.float32)

    # Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi = cv2.resize(roi, (224, 224))
    images.append(roi)

    # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    if len(roi.shape) == 2:
        roi = np.expand_dims(roi, axis=0)
    else:
        roi = np.rollaxis(roi, 2, 0)

    # Make it a batch of length 1
    roi = np.expand_dims(roi, axis=0)

    #  Make facial keypoint predictions using your loaded, trained network
    ## perform a forward pass to get the predicted facial keypoints
    roi = torch.from_numpy(roi).type(torch.FloatTensor)
    results = net.forward(roi)

    print(results.size())

    results = results.view(results.size()[0], 68, -1)
    keypoints.append(results[0])

# Display each detected face and the corresponding keypoints
visualize_output(images, keypoints)