###
# global constants and local path constants
###


###
# Note: please add path related constants definition below for keeping the code shorter in the blocks
###
# The base dir which contains the raw data. We will also save the preprocessed result in this directory
# DATA_BASE_DIR

# system path in preprocessing_training_data
# SYS_DIR

# tensorboard log path
# DETECTOR_NET_TENSORBOARD_LOG_DIR
# CLASSIFIER_NET_TENSORBOARD_LOG_DIR

# sample images folder
# DATA_SAMPLE_DIR


# Please add your favor path below with the switch
ENGINEER_FAVOR = "JM"

if ENGINEER_FAVOR == "XG":
    DATA_BASE_DIR = "/home/xuan/lung_cancer_data/"
    SYS_DIR = "/home/xuan/AIHealth"
    DETECTOR_NET_TENSORBOARD_LOG_DIR = "/home/xuan/tensorboard/detector_net"
    CLASSIFIER_NET_TENSORBOARD_LOG_DIR = "/home/xuan/tensorboard/classifier_net"
    DATA_SAMPLE_DIR = "/home/xuan/lung_cancer_data/sample_images"

if ENGINEER_FAVOR == "LH":
    DATA_BASE_DIR = "/media/luhui/E/data/lung_data/"
    SYS_DIR = "/media/luhui/E/AIDev/AIHealth"
    DETECTOR_NET_TENSORBOARD_LOG_DIR = "/media/luhui/E/data/tensorboard/0"
    CLASSIFIER_NET_TENSORBOARD_LOG_DIR = "/media/luhui/E/data/tensorboard/10"
    DATA_SAMPLE_DIR = "/media/luhui/E/data/lung_data/sample_images"

if ENGINEER_FAVOR == "JM":
    DATA_BASE_DIR = "/media/jun/Data/KaggleData/"
    SYS_DIR = "/home/jun/Desktop/workspace/erudite/AIHealth/"
    DETECTOR_NET_TENSORBOARD_LOG_DIR = "/home/jun/Desktop/workspace/erudite/tensorboard/detector_net"
    CLASSIFIER_NET_TENSORBOARD_LOG_DIR = "/home/jun/Desktop/workspace/erudite/tensorboard/classifier_net"
    DATA_SAMPLE_DIR = "/media/jun/Data/KaggleData/sample_images"



###
# global common constants
###
KAGGLE_TRAIN_DATA = "kaggle_train_data.npy"
KAGGLE_VALIDATE_DATA = "kaggle_validate_data.npy"

LUNA_TRAIN_DATA = "luna_train_data.npy"
LUNA_VALIDATE_DATA = "luna_validate_data.npy"

# TF dimension for image sizes, put in a single place for easily managing all
DIMEN_X = 96  # input dimension size, changed from 128 to 96
DIMEN_Y = 24  # output dimension size, changed from 32 to 24

# For split the image
MARGIN = 32
SIDE_LEN = 32


