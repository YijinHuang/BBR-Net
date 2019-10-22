import os

import pickle
import torch
import numpy as np
import torchvision.models as models

from config import CONFIG
from model import MyModel, MyEfficientNet
from train import train, evaluate
from data_utils import generate_data

torch.set_num_threads(8)


def main():
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

    import bbr_net
    model = MyModel(bbr_net.resnet_bbr, CONFIG['BOTTLENECK_SIZE'], CONFIG['NUM_CLASS'], pretrained=False).cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # train
    model, record_epochs, accs, losses = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=CONFIG['EPOCHS'],
        learning_rate=CONFIG['LEARNING_RATE'],
        batch_size=CONFIG['BATCH_SIZE'],
        save_path=CONFIG['SAVE_PATH']
    )
    pickle.dump(
        (record_epochs, accs, losses),
        open(CONFIG['RECORD_PATH'], 'wb')
    )

    # test
    evaluate(CONFIG['SAVE_PATH'], test_dataset, '../../result/rectify_net/origin_test')


if __name__ == '__main__':
    main()
