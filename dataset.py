import glob
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import cv2

from data_load import FacialKeypointsDataset

class Dataset():

    def __init__(self, root_dir=None, csv_file=None, training_dir=None):
        self.data_root_dir = root_dir or './data/'
        self.training_csv = os.path.join(self.data_root_dir, 'training_frames_keypoints.csv')
        self.training_data_dir = os.path.join(self.data_root_dir, 'training/')
        self.key_pts_frame = pd.read_csv(self.training_csv)
        self.face_dataset = FacialKeypointsDataset(csv_file=self.training_csv,
                                                   root_dir=self.training_data_dir)
        self.face_dataset_len = len(self.face_dataset)

    def get_image_name(self, index):
        """Get image full path"""
        image_name = self.key_pts_frame.iloc[index, 0]
        # return os.path.join(self.training_data_dir, image_name)
        return image_name

    def get_image_key_pts(self, index):
        """Get image full path"""
        key_pts = self.key_pts_frame.iloc[index, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        return key_pts

    def get_image(self, image_name):
        """Get image"""
        return mpimg.imread(os.path.join(self.training_data_dir, image_name))

    def show_image_data(self, index, data):
        """Show image info"""
        # print out some stats about the data
        print('Number of images: ', self.key_pts_frame.shape[0])

        index = index or np.random.randint(0, self.face_dataset_len)
        image_name = self.get_image_name(index)
        key_pts = self.get_image_key_pts(index)

        print('Image name: ', image_name)
        print('Landmarks shape: ', key_pts.shape)
        print('First 4 key pts: {}'.format(key_pts[:4]))
        if data:
            image_data = {'image': self.get_image(image_name), 'keypoints': key_pts}
            self.show_keypoints(image_data['image'], image_data['keypoints'])
            return image_data

    def show_keypoints(self, image, key_pts):
        """Show image with keypoints"""
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
        plt.show()

