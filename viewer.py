#!/usr/bin/env python

import os
import re
import sys
import cmd
import json
import signal
import array
import readline
import numpy
import argparse

from dataset import Dataset
from matplotlib import pyplot as plt

# if __name__ == '__main__':
#     # parse command line arguments
#     parser = argparse.ArgumentParser(
#         description='Perform ' + sys.argv[0] + ' show a smaple training image and info')
#     parser.add_argument('-i', '--image_index', type=int, help='image id', default=None)
#     parser.add_argument('-d', '--display_image', type=bool, help='image id', default=True)
#     parser.add_argument('-t', '--display_transforms', type=int, help='image id', default=None)
#     args = parser.parse_args()

#     dataset = Dataset()
#     if args.image_index:
#         dataset.show_image_data(index=args.image_index, data=args.display_image)

#     if args.display_transforms:
#         dataset.show_transformed(index=args.display_transforms)

class CmdProcessor(cmd.Cmd):

    intro = """
    Available commands:

    show                    Show an image.
    show_trans              Show a transformed image.

    help <ls/execute/... >  Help for relevant command
    """
    prompt = 'Dataset> '

    def emptyline(line):

        return

    def do_show(self, line):
        """
        Show an image with optional arguments
        show image index 1: display image by index
        """
        index = line


    def do_show_trans(self, line):
        """
        Show an transformed image with optional arguments
        show transformed image index 1: display transformed image by index
        """
        index = line

    def do_exit(self, line):
        'Exit'

        print ""
        readline.write_history_file(history_file)
        return True

    do_EOF = do_exit


if __name__ == '__main__':

    signal.signal(signal.SIGINT, SafeExit)
    signal.signal(signal.SIGTERM, SafeExit)
    signal.signal(signal.SIGHUP, SafeExit)

    try:
        if not os.path.isfile(history_file):
            os.system("touch " + history_file)
        readline.read_history_file(history_file)
    except Exception as e:
        pass

    cmd_processor = CmdProcessor()
    cmd_processor.cmdloop()


    readline.write_history_file(history_file)