import sys
import numpy
import keyboard  # using module keyboard
import argparse

from dataset import Dataset
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Perform ' + sys.argv[0] + ' show a smaple training image and info')
    parser.add_argument('-i', '--image_index', type=int, help='image id', default=None)
    parser.add_argument('-d', '--display_image', type=bool, help='image id', default=True)
    parser.add_argument('-t', '--display_transforms', type=int, help='image id', default=None)
    args = parser.parse_args()

    dataset = Dataset()
    if args.image_index:
        dataset.show_image_data(index=args.image_index, data=args.display_image)

    if args.display_transforms:
        dataset.show_transformed(index=args.display_transforms)
