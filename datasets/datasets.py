from PIL import Image, ImageOps
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class SingleDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None, max_class=None, class_bias=0):
        self.transform = transform
        self.X = []
        self.y = []

        with open(label_path, 'r') as f:
            for line in f:
                fn = line.split(' ')[0]
                y_t = line.split(' ')[1]
                y_t = int(y_t) - class_bias
                if max_class != None and y_t >= max_class:
                    continue
                self.X.append(os.path.join(img_path, fn))
                self.y.append(y_t)

    def __len__(self):
        """Reflect amount of available examples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get single image/label pair."""
        img = cv2.imread(self.X[idx])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


class PairDataset(Dataset):
    def __init__(self, src_dataset, tgt_dataset):
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.y_set = set(np.array(self.src_dataset.y))
        self.src_indices = {label: np.where(np.array(self.src_dataset.y) == label)[0]
                            for label in self.y_set}
        self.tgt_indices = {label: np.where(np.array(self.tgt_dataset.y) == label)[0]
                            for label in self.y_set}

    def __getitem__(self, idx):
        x_1, y_1 = self.src_dataset[idx]
        p_idx = np.random.choice(self.tgt_indices[y_1])
        n_y = np.random.choice(list(self.y_set - set([y_1])))
        n_idx = np.random.choice(self.src_indices[n_y])
        x_2, y_2 = self.tgt_dataset[p_idx]
        x_3, y_3 = self.src_dataset[n_idx]

        return {'anchor': x_1, 'positive': x_2, 'negative': x_3}, \
                {'anchor': y_1, 'positive': y_2, 'negative': y_3}

    def __len__(self):
        return len(self.src_dataset)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indicecs_count = {label: 0 for label in self.labels_set}


class DomainNetDataset(Dataset):
    """
    Designed for DomainNet dataset
    """
    def __init__(self, root_path, label_path, classes, transform=None):
        self.transform = transform
        self.X = []
        self.y = []

        c_idxs = {}
        idx = 0
        for c in classes:
            c_idxs[c] = idx
            idx += 1

        with open(label_path, 'r') as f:
            for line in f:
                fn, y_t = line.split(' ')
                y_t = int(y_t)
                if y_t in c_idxs:
                    self.X.append(os.path.join(root_path, fn))
                    self.y.append(c_idxs[y_t])
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.X[idx])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


class MergedDataset(Dataset):
    """
    Merge the divided dataset (a, b)
    """
    def __init__(self, a_img_path, b_img_path, a_label_path, b_label_path, transform=None):
        self.transform = transform
        self.X = []

        with open(a_label_path, 'r') as f:
            for line in f:
                fn = line.split(' ')[0]
                self.X.append(os.path.join(a_img_path, fn))

        with open(b_label_path, 'r') as f:
            for line in f:
                fn = line.split(' ')[0]
                self.X.append(os.path.join(b_img_path, fn))
    
    def __len__(self):
        """Reflect amount of available examples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get single image/label pair."""
        img = Image.open(self.X[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img


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

    if domain == 'N':
        transform_list.append(Negative())

    transform_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)
    
def get_officeHome_transform(train=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda x: x.convert("RGB")))
    if train:
        transform_list.append(transforms.RandomResizedCrop(224))
        transform_list.append(transforms.RandomHorizontalFlip())
    else:
        transform_list.append(transforms.Resize(224))
        # transform_list.append(transforms.CenterCrop(224))

    transform_list += [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    return transforms.Compose(transform_list)