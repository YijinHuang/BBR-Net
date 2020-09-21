import os
import sys
import csv

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

from tqdm import tqdm
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(tuple(MEAN), tuple(STD)),
])


class RefineNet():
    def __init__(self, model_path):
        self.model_path = model_path

    def initial(self, csv_path):
        self.model = torch.load(self.model_path).eval().cuda()
        self.preprocess = test_preprocess
        self.annotations = self.read_csv(csv_path)
        self.new_annotations = {}

    def batch_refine(self):
        for file_path in tqdm(self.annotations.keys()):
            torch.set_grad_enabled(False)
            image = Image.open(file_path)
            for coor in self.annotations[file_path]:
                x1, y1, x2, y2 = coor
                w = x2 - x1
                h = y2 - y1
                lesion_patch = image.crop((x1, y1, x2, y2))

                if file_path not in self.new_annotations.keys():
                    self.new_annotations[file_path] = []

                new_coor = self.refine(lesion_patch)
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = new_coor

                new_x1 = int(w * x1_ratio + x1)
                new_y1 = int(h * y1_ratio + y1)
                new_x2 = int(w * x2_ratio + x1)
                new_y2 = int(h * y2_ratio + y1)
                if new_x1 == new_x2:
                    new_x1, new_x2 = x1, x2
                if new_y1 == new_y2:
                    new_y1, new_y2 = y1, y2
                self.new_annotations[file_path].append((new_x1, new_y1, new_x2, new_y2))

    def write_annotations(self, new_csv_path, cat):
        with open(new_csv_path, 'w') as csv_file:
            for file_path, coors in self.new_annotations.items():
                for coor in coors:
                    x1, y1, x2, y2 = coor
                    csv_file.write('{},{},{},{},{},{}\n'.format(file_path, x1, y1, x2, y2, cat))

    def refine(self, lesion_patch):
        patch = self.preprocess(lesion_patch).unsqueeze(0).cuda()
        pred = self.model(patch)
        return pred.cpu().numpy().tolist()[0]

    def read_csv(self, csv_path):
        annotations = {}
        with open(csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                img_file, x1, y1, x2, y2 = row[:5]

                if img_file not in annotations.keys():
                    annotations[img_file] = []

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                annotations[img_file].append((x1, y1, x2, y2))
            return annotations


# inference example
if __name__ == "__main__":
    trained_model_path = 'path/to/model'
    origin_annotation = 'path/to/origin_annotation_csv_file'
    refined_annotation = 'path/to/refined_annotation_csv_file'
    lesion_name = 'HEM'

    refine_net = RefineNet(trained_model_path)
    refine_net.initial(origin_annotation)
    refine_net.batch_refine()
    refine_net.write_annotations(refined_annotation, lesion_name)
