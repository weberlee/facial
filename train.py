import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import Net
net = Net()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_tools import DataTools

data_tools = DataTools()
transformed_training_dataset = data_tools.transformed_training_data

# load training data in batches
batch_size = 10
train_loader = DataLoader(transformed_training_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

# Define the loss and optimization
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

def train_net(n_epochs):

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')

# train your network
n_epochs = 4 # start small, and increase when you've decided on your model structure and hyperparams
train_net(n_epochs)

# ## TODO: change the name to something uniqe for each new model
model_dir = 'saved_models/'
model_name = 'keypoints_model_1.pt'

# # after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)
