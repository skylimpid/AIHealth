from Training.prepare import full_prep, prepare_luna, preprocess_luna
from Net.tensorflow_model.DetectorNet import DecetorNet, get_model
from Training.configuration_training import cfg
from Training.Detector.TrainingDetectorData import TrainingDetectorData
import os
import time

# start prepare the trainning data

print("Start to prepare the trainning data.")

print("Preprocessing the kaggle data")

full_prep(step1=True, step2=True)

print("Preprocessing the luna data")
prepare_luna()
preprocess_luna()
print("Finish the preprocessing")

net_config, net, loss, get_pbb = get_model()


save_dir = cfg.DIR.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logfile = os.path.join(save_dir,'log')

datadir = cfg.DIR.preprocess_result_path

dataset = TrainingDetectorData(
    datadir,
    '/Users/xuan/lung_cancer_data/full.npy',
    net_config,
    phase='train')

start_time = time.time()
