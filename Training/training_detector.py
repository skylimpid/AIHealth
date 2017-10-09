from Training.prepare import full_prep, prepare_luna, preprocess_luna
from Net.tensorflow_model.DetectorNet import DecetorNet, get_model
from Training.configuration_training import cfg
from Training.Detector.TrainingDetectorData import TrainingDetectorData
import os
import time
import tensorflow as tf

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



imgs, bbox, coord = dataset.__getitem__(0)

print(imgs.shape)
print(bbox.shape)
print(coord.shape)

import sys

sys.exit(0)

X = tf.placeholder(tf.float32, shape=(None, 128, 128, 128, 1))
coord = tf.placeholder(tf.float32, shape=(None, 32, 32, 32, None))

feat, out = net.getDetectorNet(X, coord)

# optimizer and learning rate
global_step = tf.Variable(0, trainable=False)

lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                cfg.TRAIN.EPOCHS, 0.1, staircase=True)
monentum = cfg.TRAIN.MOMENTUM

train_op = tf.train.MomentumOptimizer(lr, monentum).minimize(loss, global_step=global_step)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #initialize variables
    sess.run(init)
    for step in range(1, cfg.TRAIN.EPOCHS):
        pass
