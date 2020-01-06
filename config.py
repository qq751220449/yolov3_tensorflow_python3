# coding:utf-8

# yolo
TRAIN_INPUT_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640 ,672, 704, 736, 768, 800, 832, 864, 896, 928]
TEST_INPUT_SIZE = 800
# TEST_INPUT_SIZE = 544
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 2
LEARN_RATE_INIT = 1e-4
LEARN_RATE_END = 1e-6
WARMUP_PERIODS = 2
MAX_PERIODS = 100 

GT_PER_GRID = 3
MOVING_AVE_DECAY = 0.9995
# SCORE_THRESHOLD = 0.01    # The threshold of the probability of the classes

# test
SCORE_THRESHOLD = 0.6   # The threshold of the probability of the classes
IOU_THRESHOLD = 0.45     # The threshold of the IOU when implement NMS
WEIGHT_FILE_TEST = 'checkpoint/yolo.ckpt-99-0.7929'    # The weight file for Test

# name and path
DATASET_PATH = '/home/ice2019/yolov3_temsorflow/data'
PROJECT_PATH = '/home/ice2019/yolov3_temsorflow'
ANNOT_DIR_PATH = 'data/VOC2007'                    # 转换后的Ann文件存放目录
WEIGHTS_DIR = 'weights'
WEIGHTS_INIT = 'darknet2tf/saved_model/darknet53.ckpt'
LOG_DIR = 'log'
CLASSES = ['bottle', 'plastic_case', 'trash_bag']

# Continue To Train
# Continue_To_Train = False
Continue_To_Train = True
CHECKPOINT_FILE = 'checkpoint/yolo.ckpt-99-0.7929'

# ckpt to pb file
CKPT2PB_CKPT_FILE = 'checkpoint/yolo.ckpt-99-0.7929'
CKPT2PB_PB_NAME = 'frozen_model_20200103_99_7929.pb'
