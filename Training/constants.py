## define constants

###
# add environment related configuration constants here for easy to manage instead of hard coding randomly
###

# The base dir which contains the raw data. We will also save the preprocessed result in this directory
DATA_BASE_DIR = "/home/xuan/lung_cancer_data/"

# system path in preprocessing_training_data
SYS_DIR = "/home/xuan/AIHealth"

# tensorboard log path
DETECTOR_NET_TENSORBOARD_LOG_DIR = "/home/xuan/tensorboard/detector_net"
CLASSIFIER_NET_TENSORBOARD_LOG_DIR = "/home/xuan/tensorboard/classifier_net"

# sample images folder
DATA_SAMPLE_DIR = "/home/xuan/lung_cancer_data/sample_images"

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


