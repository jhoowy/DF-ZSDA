import os
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np

from color_domain import ColorConverter

import argparse

"""
TODO:
    - Refactor to save dataset as hdf5 format
"""

class Canny(object):
    def __init__(self, minVal=50, maxVal=200):
        assert isinstance(minVal, (int, float))
        assert isinstance(maxVal, (int, float))
        self.minVal = minVal
        self.maxVal = maxVal

    def __call__(self, img):
        img = np.array(img)
        img = cv2.Canny(img, self.minVal, self.maxVal)
        img = Image.fromarray(img)
        return img


class Negative(object):
    def __call__(self, img):
        img = ImageOps.invert(img)
        return img


def get_transform(domain='G', resize_dim=128):
    """
    Parameters:
        domain (str)        -- the name of target domain
        resize_dim (int)    -- resize input image to this size
    """
    transform_list = []
    transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))
    
    if resize_dim is not None:
        transform_list.append(transforms.Resize(resize_dim))

    if domain == 'E':
        transform_list.append(Canny())
        transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))
    elif domain == 'N':
        transform_list.append(Negative())

    return transforms.Compose(transform_list)


class DomainTransformer:

    def __init__(self, color_src, resize_dim=None):
        self.color_transformer = ColorConverter(color_src)
        self.resize_dim = resize_dim

    def save_color(self, dataset, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        label_path = '_'.join([save_dir, 'labels.txt'])
        transform = get_transform(resize_dim=self.resize_dim)
        with open(label_path, 'w') as label_file:
            idx = 0
            for img, label in dataset:
                img = self.color_transformer.convert(img)
                img = transform(img)
                img_name = "%08d.png" % idx
                img.save(os.path.join(save_dir, img_name))
                label_file.write(img_name + " " + str(int(label)) + "\n")
                idx += 1
    
    def save_original(self, dataset, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        label_path = '_'.join([save_dir, 'labels.txt'])
        transform = get_transform(resize_dim=self.resize_dim)
        with open(label_path, 'w') as label_file:
            idx = 0
            for img, label in dataset:
                img = transform(img)
                img_name = "%08d.png" % idx
                img.save(os.path.join(save_dir, img_name))
                label_file.write(img_name + " " + str(int(label)) + "\n")
                idx += 1

    def save_edge(self, dataset, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        label_path = '_'.join([save_dir, 'labels.txt'])
        transform = get_transform('E', resize_dim=self.resize_dim)
        with open(label_path, 'w') as label_file:
            idx = 0
            for img, label in dataset:
                img = transform(img)
                img_name = "%08d.png" % idx
                img.save(os.path.join(save_dir, img_name))
                label_file.write(img_name + " " + str(int(label)) + "\n")
                idx += 1
        return
    
    def save_negative(self, dataset, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        label_path = '_'.join([save_dir, 'labels.txt'])
        transform = get_transform('N', resize_dim=self.resize_dim)
        with open(label_path, 'w') as label_file:
            idx = 0
            for img, label in dataset:
                img = transform(img)
                img_name = "%08d.png" % idx
                img.save(os.path.join(save_dir, img_name))
                label_file.write(img_name + " " + str(int(label)) + "\n")
                idx += 1
        return


def load_dataset(name, dir):
    if name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(dir, train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(dir, train=False, download=True)
    elif name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(dir, train=True, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(dir, train=False, download=True)
    elif name == 'EMNIST':
        train_dataset = torchvision.datasets.EMNIST(dir, split='letters', train=True, download=True)
        test_dataset = torchvision.datasets.EMNIST(dir, split='letters', train=False, download=True)
    else:
        raise NotImplementedError('Dataset {} is not supported'.format(name))

    return train_dataset, test_dataset
    

def main(args):
    bsds_dir = os.path.join(args.root, args.bsds_dir)
    dataset_dir = os.path.join(args.root, args.name)
    domain_transformer = DomainTransformer(bsds_dir)

    train_dataset, test_dataset = load_dataset(args.name, args.root)
    dirs = [dataset_dir, dataset_dir + '_test']
    datasets = [train_dataset, test_dataset]

    for save_dir, dataset in zip(dirs, datasets):
        os.makedirs(save_dir, exist_ok=True)
        domain_transformer.save_original(dataset, os.path.join(save_dir, 'original'))
        domain_transformer.save_color(dataset, os.path.join(save_dir, 'color'))
        domain_transformer.save_edge(dataset, os.path.join(save_dir, 'edge'))
        domain_transformer.save_negative(dataset, os.path.join(save_dir, 'negative'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='FashionMNIST', help='name of the dataset. [MNIST | FashionMNIST | EMNIST]')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--bsds_dir', type=str, default='BSDS500')
    args = parser.parse_args()

    main(args)
