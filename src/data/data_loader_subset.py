"""
Adaptation of data_loader.py to generate a data loader from 10 selected subclasses of CIFAR-100.
The subclasses are specified in the parameter TL_label_map, a dictionary mapping the selected CIFAR-100 class numbers
to their matching CIFAR-10 class number.
The code loosely based on: https://github.com/MadryLab/cifar10_challenge
"""
import os
import torch
from torch.utils import data
from PIL import Image, ImageFile
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class ImgDataset(data.Dataset):
    def __init__(self, data_dir, batches, max_nr_img,
                 transform=None, label_map = None):
        self.image_paths = []
        if label_map != None:
            label_map_copy = label_map.copy()
            label_map_keys = list(label_map_copy.keys())
            for label in label_map_keys:
                label_map_copy[int(label)] = label_map_copy.pop(label)
            self.label_map = label_map_copy
        else:
            self.label_map = {i:i for i in range(0,100)}

        self.transform = transform
        batch_paths = [os.path.join(data_dir, batchname) for batchname in batches]
        self.max_nr_img = max_nr_img
        self._read_images(batch_paths)

    
    def _read_images(self, data_dir):
        """ Reads images from directory."""
        nr_img = 0
        for data_batch in data_dir:
            if nr_img <= self.max_nr_img:
                batch = unpickle(data_batch)
                labels = batch[b'labels']
                image_data = batch[b'data']
                image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
                for i in range(10000):
                    if nr_img < self.max_nr_img and labels[i] in self.label_map.keys():
                        self.image_paths.append({
                        'class': self.label_map[labels[i]],
                        'img': image_data[i]
                        })
                        nr_img += 1

    
    def __len__(self):
        """ Returns the total number of images."""
        return len(self.image_paths)

    
    def __getitem__(self, index):
        """Gets one data sample."""
        img = self.image_paths[index]
        target = torch.tensor(img['class'])
        image = Image.fromarray(img['img'])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_file_name(file_path):
  """Returns file name when given a full file path."""
    return file_path.split('/')[-1]


def list_content(path):
    """Lists all files/folders in a specific directory."""
    return [os.path.join(path, x) for x in os.listdir(path)]


def read_dataset(data_dir, mode, max_nr_img, transform=None, run_id = None, label_map=None):
    """Returns training and validation set."""
    if mode == 'train':
        batches = ['data_batch_{}_balanced'.format(ii + 1) for ii in range(4)]
        dataset = ImgDataset(data_dir, batches, max_nr_img, transform=transform, label_map=label_map)

    elif mode == 'val':
        batches = ['data_batch_5_balanced']
        dataset = ImgDataset(data_dir, batches, max_nr_img, transform=transform, label_map=label_map)

    elif mode == 'test':
        batches = ['test_batch']
        dataset = ImgDataset(data_dir, batches, max_nr_img, transform=transform, label_map=label_map)

    return dataset


def get_loader(mode, data_dir, transform, batch_size, shuffle, num_workers, max_nr_img = 50000, run_id=None,
               label_map = None):
    dataset = read_dataset(data_dir, mode, max_nr_img, transform = transform, run_id = run_id, label_map=label_map)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #drop_last=True,
                                              pin_memory=True)
    return data_loader
