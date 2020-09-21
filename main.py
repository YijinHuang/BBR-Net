import os
import pickle
import random

import torch
import numpy as np
import torchvision.models as models

import bbr_net
from config import CONFIG
from model import MyModel
from train import train, evaluate
from data_utils import generate_data


def main():
    # reproducibility
    seed = CONFIG['RANDOM_SEED']
    set_random_seed(seed)

    # load dataset
    train_dataset, test_dataset, val_dataset = generate_data(
        CONFIG['TRAIN_PATH'],
        CONFIG['TEST_PATH'],
        CONFIG['TEST_PATH'],
        CONFIG['INPUT_SIZE']
    )

    save_dir = os.path.split(CONFIG['SAVE_PATH'])[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = MyModel(bbr_net.resnet_bbr, CONFIG['BOTTLENECK_SIZE'], CONFIG['NUM_CLASS'], pretrained=False).cuda()

    # train
    model, record_epochs, val_ious, losses = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=CONFIG['EPOCHS'],
        learning_rate=CONFIG['LEARNING_RATE'],
        batch_size=CONFIG['BATCH_SIZE'],
        save_path=CONFIG['SAVE_PATH']
    )
    pickle.dump(
        (record_epochs, val_ious, losses),
        open(CONFIG['RECORD_PATH'], 'wb')
    )

    # test
    visualized_images_path = CONFIG['VISUALIZED_IMAGES_PATH']
    if not os.path.exists(visualized_images_path):
        os.makedirs(visualized_images_path)
    evaluate(CONFIG['SAVE_PATH'], test_dataset, visualized_images_path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
