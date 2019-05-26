#!/usr/bin/env python

import os
import re
import sys
import cmd
import json
import readline
import subprocess
import threading
import numpy

from threading import Thread
from dataset import Dataset
from matplotlib import pyplot as plt

history_file = os.path.expanduser("~/.dataset_shell.hist")

dataset = Dataset()

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
        dataset.show_image_data(index=index, data=True)


    def do_show_trans(self, line):
        """
        Show an transformed image with optional arguments
        show transformed image index 1: display transformed image by index
        """
        index = line

    def do_exit(self, line):
        'Exit'

        print("")
        readline.write_history_file(history_file)
        return True

    do_EOF = do_exit


if __name__ == '__main__':

    try:
        if not os.path.isfile(history_file):
            os.system("touch " + history_file)
        readline.read_history_file(history_file)
    except Exception as e:
        pass

    cmd_processor = CmdProcessor()
    cmd_processor.cmdloop()


    readline.write_history_file(history_file)