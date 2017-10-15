import sys
from Preprocessing.preprocessing import full_prep, preprocess_luna, prepare_luna
from Training.configuration_training import cfg

sys.path.append("/Users/xuan/AIHealth")


def main():
    # start prepare the trainning data

    print("Start to prepare the trainning data.")

    print("Preprocessing the kaggle data")

    full_prep(cfg, step1=True, step2=True)

    print("Preprocessing the luna data")
    prepare_luna(cfg)
    preprocess_luna(cfg)
    print("Finish the preprocessing")



if __name__ == "__main__":
    main()