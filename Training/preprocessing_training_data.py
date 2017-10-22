import sys

from Preprocessing.preprocessing import full_prep, preprocess_luna, prepare_luna
from Training.configuration_training import cfg
from Training.constants import SYS_DIR
import os

sys.path.append(SYS_DIR)


def main():
    # start prepare the training data
    print("Start to prepare the training data.")

    if not os.path.exists(cfg.DIR.train_split_data_path):
        os.makedirs(cfg.DIR.train_split_data_path)

    print("Preprocessing the kaggle data")

    full_prep(cfg, step1=True, step2=True)

    print("Preprocessing the luna data")
    prepare_luna(cfg)
    preprocess_luna(cfg)
    print("Finish the preprocessing")


if __name__ == "__main__":
    main()