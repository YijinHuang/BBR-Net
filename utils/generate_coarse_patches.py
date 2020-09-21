import os
import csv
import random

import cv2 as cv
from tqdm import tqdm


annotation_file = '/path/to/full/image/annotation/csv/file'
img_save_path = '/folder/to/save/simulated/coarse/annotations/patches'
csv_save_path = '/path/to/save/patches/annotation'
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

random.seed(10)


def read_csv():
    with open(annotation_file, 'r') as file:
        reader = csv.reader(file)

        result = {}
        for row in reader:
            img_file, x1, y1, x2, y2 = row[:5]

            if img_file not in result:
                result[img_file] = []

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

    return result


def crop_img(result, repeat=5):
    with open(csv_save_path, 'a') as annotation_csv:
        for img_name, annotations in tqdm(result.items()):
            img_pre = os.path.splitext(os.path.split(img_name)[-1])[0]
            img = cv.imread(img_name)
            img_shape = img.shape
            for a in annotations:
                x1 = int(a['x1'])
                x2 = int(a['x2'])
                y1 = int(a['y1'])
                y2 = int(a['y2'])

                for _ in range(repeat):
                    bbox = [x1, y1, x2, y2]

                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]

                    if w * h > 6400:
                        w_expand_ratio = random.random() * 0.4 + 0.1
                        h_expand_ratio = random.random() * 0.4 + 0.1
                        w_expand = w_expand_ratio * w
                        h_expand = h_expand_ratio * h

                        w_shift_ratio = random.random()
                        h_shift_ratio = random.random()
                        left_x_shift = w_shift_ratio * w_expand
                        right_x_shift = (1 - w_shift_ratio) * w_expand
                        top_y_shift = h_shift_ratio * h_expand
                        bottom_y_shift = (1 - h_shift_ratio) * h_expand

                        bbox[0] = int(max((0, bbox[0] - left_x_shift)))
                        bbox[1] = int(max((0, bbox[1] - top_y_shift)))
                        bbox[2] = int(min((bbox[2] + right_x_shift, img_shape[1])))
                        bbox[3] = int(min((bbox[3] + bottom_y_shift, img_shape[0])))

                        new_x1, new_y1, new_x2, new_y2 = bbox
                        new_w = new_x2 - new_x1
                        new_h = new_y2 - new_y1
                        rl_x1 = (x1 - new_x1) / new_w
                        rl_x2 = (x2 - new_x1) / new_w
                        rl_y1 = (y1 - new_y1) / new_h
                        rl_y2 = (y2 - new_y1) / new_h

                        crop_name = '{}_{}_{}_{}_{}.jpg'.format(img_pre, new_x1, new_y1, new_x2, new_y2)
                        crop_path = os.path.join(img_save_path, crop_name)
                        annotation_csv.write('{},{},{},{},{}\n'.format(crop_path, rl_x1, rl_y1, rl_x2, rl_y2))
                        cv.imwrite(crop_path, img[new_y1:new_y2, new_x1:new_x2])


if __name__ == "__main__":
    result = read_csv()
    crop_img(result)
