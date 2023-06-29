"""
Implements data_loaders for CIFAR-10, CIFAR-100, SVHN and MNIST.
Code loosely based on: https://github.com/MadryLab/cifar10_challenge.
"""

import os
import torch
from torch.utils import data
from PIL import Image, ImageFile
import scipy.io as sio

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImgDataset(data.Dataset):
    # Initialization
    def __init__(self, data_dir, batches, max_nr_img,
                 transform=None):
        self.image_paths = []
        self.transform = transform
        batch_paths = [os.path.join(data_dir, batchname) for batchname in batches]
        self.max_nr_img = max_nr_img
        self._read_images(batch_paths)

    # Read images from directory
    def _read_images(self, data_dir):
        nr_img = 0
        for data_batch in data_dir:
            if nr_img <= self.max_nr_img:
                batch = unpickle(data_batch)
                labels = batch[b'labels']
                image_data = batch[b'data']
                image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
                for i in range(10000):
                    if nr_img < self.max_nr_img:
                        self.image_paths.append({
                        'class': labels[i],
                        'img': image_data[i]
                        })
                        nr_img += 1

    # Returns the total number of images
    def __len__(self):
        return len(self.image_paths)

    # get one data sample
    def __getitem__(self, index):
        img = self.image_paths[index]
        target = torch.tensor(img['class'])
        image = Image.fromarray(img['img'])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


import numpy as np
class NpyDataset(data.Dataset):
    # Initialization
    def __init__(self, npy_path, run_id, max_nr_img, adv,
                 transform=None):
        self.image_paths = []
        self.transform = transform
        self.npy_path = npy_path
        self.max_nr_img = max_nr_img
        self._read_images(npy_path, run_id, adv)

    # Read images from directory
    def _read_images(self, npy_path, run_id, adv):
        nr_img = 0
        if adv:
            np_images = np.load(npy_path + '_x' + str(run_id) + '.npy')
        else:
            np_images = np.load(npy_path + 'nat_x' + str(run_id) + '.npy')
        np_labels = np.load(npy_path + '_labels' + str(run_id) + '.npy')
        np_images = np_images.transpose(0, 2, 3, 1)
        for i in range(10000):
            if nr_img < self.max_nr_img:
                self.image_paths.append({
                'class': np_labels[i],
                'img': np_images[i,:,:,:].astype(np.uint8)
                })
                nr_img += 1
                img_paths=self.image_paths

    # Returns the total number of images
    def __len__(self):
        return len(self.image_paths)

    # get one data sample
    def __getitem__(self, index):
        img = self.image_paths[index]
        target = torch.tensor(img['class'])
        image = Image.fromarray(img['img'])
        if self.transform is not None:
            image = self.transform(image)
        return image, target

class SVHNDataset(data.Dataset):
    # Initialization
    def __init__(self, data_dir, file_name, max_nr_img, transform=None):
        self.image_paths = []
        self.transform = transform
        data_path = os.path.join(data_dir, file_name)
        self.max_nr_img = max_nr_img
        self._read_images(data_path)

    # Read images from directory
    def _read_images(self, data_path):
        nr_img = 0
        data = sio.loadmat(data_path)
        labels = data['y']
        labels[labels == 10] = 0  # original labels given in {1, ..., 10} with label 10 corresponding to digit 0
        image_data = data['X']
        image_data = image_data.transpose(3, 0, 1, 2)
        for i in range(len(labels)):
            if nr_img < self.max_nr_img:
                self.image_paths.append({
                'class': int(labels[i]),
                'img': image_data[i]
                })
                nr_img += 1

    # Returns the total number of images
    def __len__(self):
        return len(self.image_paths)

    # get one data sample
    def __getitem__(self, index):
        img = self.image_paths[index]
        target = torch.tensor(img['class'])
        image = Image.fromarray(img['img'])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class MNISTDataset(data.Dataset):
    # Initialization
    def __init__(self, data_dir, file_name, max_nr_img, transform=None):
        self.image_paths = []
        self.transform = transform
        data_path = os.path.join(data_dir, file_name)
        self.max_nr_img = max_nr_img
        self._read_images(data_path)

    # Read images from directory
    def _read_images(self, data_path):
        nr_img = 0
        data = sio.loadmat(data_path)
        labels = data['y'].flatten()
        image_data = data['X']
        for i in range(len(labels)):
            if nr_img < self.max_nr_img:
                self.image_paths.append({
                'class': labels[i],
                'img': np.pad(image_data[i], 2),  # pad image to resize from 28x28 to 32x32
                })
                nr_img += 1

    # Returns the total number of images
    def __len__(self):
        return len(self.image_paths)

    # get one data sample
    def __getitem__(self, index):
        img = self.image_paths[index]
        target = torch.tensor(img['class'])
        image = Image.fromarray(img['img']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# return file name when given a full file path
def get_file_name(file_path):
    return file_path.split('/')[-1]


#list all files/folders in a specific directory
def list_content(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

# return training and validation set
def read_dataset(data_dir, mode, max_nr_img, transform=None, data_set=False, run_id = None):
    if data_set == 'SVHN':
        file_name = mode + '.mat'
        dataset = SVHNDataset(data_dir, file_name, max_nr_img, transform=transform)

    elif data_set == 'MNIST':
        file_name = mode + '.mat'
        dataset = MNISTDataset(data_dir, file_name, max_nr_img, transform=transform)

    else:
        if mode == 'train':
            batches = ['data_batch_{}_balanced'.format(ii + 1) for ii in range(4)]
            dataset = ImgDataset(data_dir, batches, max_nr_img, transform=transform)

        elif mode == 'val':
            batches = ['data_batch_5_balanced']
            dataset = ImgDataset(data_dir, batches, max_nr_img, transform=transform)

        elif mode == 'test':
            batches = ['test_batch']
            dataset = ImgDataset(data_dir, batches, max_nr_img, transform=transform)

        elif mode == 'adv':
            adv = True
            dataset = NpyDataset(data_dir, run_id, max_nr_img, adv, transform=transform)

        else:
            adv = False
            dataset = NpyDataset(data_dir, run_id, max_nr_img, adv, transform=transform)

    return dataset

def get_loader(mode, data_dir, transform, batch_size, shuffle, num_workers, max_nr_img = 50000, data_set=False, run_id=None):
    dataset = read_dataset(data_dir, mode, max_nr_img, transform = transform, data_set=data_set, run_id = run_id)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True)
    return data_loader
