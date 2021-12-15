import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

from image_folder import ImageFolder

class ColorConverter:

    def __init__(self, path, shuffle=True):
        self.color_dataset = ImageFolder(path)
        self.shuffle = shuffle
        self.index = 0
        if self.shuffle:
            self.it = np.random.permutation(len(self.color_dataset))
        else:
            self.it = np.arange(len(self.color_dataset))

    def convert(self, img):
        img = img.convert('RGB')
        color_src = transforms.RandomCrop(img.size, pad_if_needed=True)(self.color_dataset[self.index])
        color_img = cv2.absdiff(np.asarray(img), np.asarray(color_src))
        color_img = Image.fromarray(color_img)

        self.index += 1
        if self.index >= len(self.color_dataset):
            self.index = 0
            if self.shuffle:
                self.it = np.random.permutation(len(self.color_dataset))

        return color_img