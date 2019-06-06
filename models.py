## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output = (W-F)/S+1 = (224-5)/1+1=220, the output tensor for one image will be (32, 220, 220)

        # Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, 5)
        # output = (W-F)/S +1 = (110-3)/1 +1 = 108
        self.conv2 = nn.Conv2d(32, 64, 3)
        # output = (W-F)/S +1 = (110-3)/1 +1 = 108
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # (32, 110, 110)
        self.pool = nn.MaxPool2d(2, stride=2)

        # Convolutional Dropout
        self.conv1_drop = nn.Dropout(p=0.1)
        self.conv2_drop = nn.Dropout(p=0.2)
        self.conv3_drop = nn.Dropout(p=0.3)
        self.conv4_drop = nn.Dropout(p=0.4)

        # Fully Connected Layer
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 136)

        # Fully Connected Dropout
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2_drop = nn.Dropout(p=0.6)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting



    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # four conv/relu + pool layers
        x = F.relu(self.pool(self.conv1(x)))
        x = self.conv1_drop(x)
        x = F.relu(self.pool(self.conv2(x)))
        x = self.conv2_drop(x)
        x = F.relu(self.pool(self.conv3(x)))
        x = self.conv3_drop(x)
        x = F.relu(self.pool(self.conv4(x)))
        x = self.conv4_drop(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)

        # three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
