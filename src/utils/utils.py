"""
Helper functions
"""

import json
from collections import OrderedDict
import numpy as np
import torch
from torch import nn as nn

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def write_json(data, file_path):
    with open(file_path, 'w') as out:
        json.dump(data, out, indent=4)


def copy_file(src_path, dest_path):
    with open(src_path) as src:
        with open(dest_path, 'w') as dest:
            dest.write(src.read())
