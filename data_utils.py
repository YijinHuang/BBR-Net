import os
import sys
import csv

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

from PIL import Image, ImageOps

# channel means and standard deviations of kaggle dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# for color augmentation, computed with make_pca.py
U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                  [-0.5989477, -0.02304967, -0.80036049],
                  [-0.56694071, -0.6935729, 0.44423429]], dtype=torch.float32)
EV = torch.tensor([1.65513492, 0.48450358, 0.1565086], dtype=torch.float32)


def generate_data(train_path, test_path, val_path, input_size):
    train_preprocess = transforms.Compose([
        # transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD)),
    ])

    test_preprocess = transforms.Compose([
        # transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD)),
    ])

    train_dataset = RectifyDataset(train_path, input_size, transform=train_preprocess)
    test_dataset = RectifyDataset(test_path, input_size, transform=test_preprocess)
    val_dataset = RectifyDataset(val_path, input_size, transform=test_preprocess)

    return train_dataset, test_dataset, val_dataset


class RectifyDataset(Dataset):
    def __init__(self, train_file, input_size, transform=None):
        self.train_file = train_file
        self.input_size = input_size
        self.transform = transform
        self.loader = default_loader

        with self._open_for_csv(self.train_file) as file:
            self.image_data = self._read_annotations(csv.reader(file, delimiter=','))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        return function(value)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_names[idx]
        annot = self.load_annotations(idx)
        img = self.loader(path)
        img, annot = self.padding_resize(img, annot)
        if self.transform is not None:
            img = self.transform(img)

        return img, annot

    def load_annotations(self, image_index):
        # get ground truth annotations
        a = self.image_data[self.image_names[image_index]]
        annotation = np.zeros(4)

        x1 = a['x1']
        x2 = a['x2']
        y1 = a['y1']
        y2 = a['y2']

        annotation[0] = x1
        annotation[1] = y1
        annotation[2] = x2
        annotation[3] = y2

        return annotation

    def _read_annotations(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            img_file, x1, y1, x2, y2 = row[:6]

            x1 = self._parse(x1, float, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, float, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, float, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, float, 'line {}: malformed y2: {{}}'.format(line))

            result[img_file] = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
        return result

    def padding_resize(self, img, annot):
        desired_size = self.input_size

        old_size = img.size

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        if old_size[1] < old_size[0]:
            new_ratio = new_size[1] / float(desired_size)
            annot[1] = new_ratio * annot[1] + ((1-new_ratio) / 2)
            annot[3] = new_ratio * annot[3] + ((1-new_ratio) / 2)
        else:
            new_ratio = new_size[0] / float(desired_size)
            annot[0] = new_ratio * annot[0] + ((1-new_ratio) / 2)
            annot[2] = new_ratio * annot[2] + ((1-new_ratio) / 2)

        img = img.resize(new_size, Image.ANTIALIAS)

        new_img = Image.new("RGB", (desired_size, desired_size))
        new_img.paste(img, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))

        self.to_center(annot)
        # new_img = img.resize((desired_size, desired_size), Image.ANTIALIAS)
        return new_img, annot

    def to_center(self, annot):
        x1, y1, x2, y2 = annot

        annot[0] = (x1+x2) / 2
        annot[1] = (y1+y2) / 2
        annot[2] = x2 - x1
        annot[3] = y2 - y1
        
    def _padding_annot(self, annot, old_length):
        new_length = self.input_size

        ratio = old_length / new_length


class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img):
        sigma = self.sigma
        if not sigma > 0.0:
            color_vec = torch.zeros(3, dtype=torch.float32)
        else:
            color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))

        color_vec = color_vec.squeeze()
        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.t())
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]


    imgs = torch.from_numpy(np.concatenate(imgs))
    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annots}


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
