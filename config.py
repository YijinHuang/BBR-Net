CONFIG = {
    'TRAIN_PATH': '../train/gt.csv',
    'TEST_PATH': '../test/gt.csv',
    'SAVE_PATH': '../result/bbr_net_giou.pt',
    'RECORD_PATH': '../result/bbr_net_giou.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 0.01,
    'INPUT_SIZE': 128,
    'BOTTLENECK_SIZE': 512,
    'NUM_CLASS': 4,
    'BATCH_SIZE': 64,
    'EPOCHS': 100,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-30, 30),
        'translation_ratio': (20 / 128, 20 / 128),  # 20 pixel in the report
        'sigma': 0.5
    }
}
