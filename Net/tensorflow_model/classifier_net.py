import tensorflow as tf
import numpy as np
from Net.tensorflow_model.detector_net import DetectorNet
from Training.configuration_training import cfg
from Training.constants import DIMEN_X, DIMEN_Y

config = {}
config['topk'] = 5
config['resample'] = None
config['preload_train'] = True
config['preload_val'] = True

config['padmask'] = False

config['crop_size'] = [DIMEN_X, DIMEN_X, DIMEN_X]
config['scaleLim'] = [0.85, 1.15]
config['radiusLim'] = [6, 100]
config['jitter_range'] = 0.15
config['isScale'] = True

config['random_sample'] = False
config['T'] = 1
config['topk'] = 5
config['stride'] = 4
config['augtype'] = {'flip': True, 'swap': True, 'rotate': True, 'scale': True}

config['detect_th'] = 0.05
config['conf_th'] = -1
config['nms_th'] = 0.05
config['filling_value'] = 160

config['startepoch'] = 20
config['lr_stage'] = np.array([50,100,140,160,180])
config['lr'] = [0.01,0.001,0.0001,0.00001,0.000001]
config['miss_ratio'] = 0.1
config['miss_thresh'] = 0.05
config['blacklist'] = {"02801e3bbcc6966cb115a962012c35df","229b8b785f880f61d8dad636c3dc2687","322bf0acacba9650fa5656b9613c75c8","627836151c555187503dfe472fb15001",
                     #  "6969c031ee0c34053faff3aac9dd2da7","9b871732b3935661e7639e84a6ab9747","aa2747369e1a0c724bea611ea7e5ffcf","c0279b938ac1790f1badfcd4459f5c4c",
                     #  "eb8d5136918d6859ca3cc3abafe369ac","f7a03adba817f2a2249b9dee0586f4be","0f5ab1976a1b1ef1c2eb1d340b0ce9c4","868b024d9fa388b7ddab12ec1c06af38",
                       "9f52323d216f89d300612cfac0122d8b","b635cda3e75b4b7238c18c6a5f1858f6"}


class ClassifierNet(object):

    DATA_FORMAT = 'channels_first'

    def __init__(self, detectorNet, img_row=DIMEN_X, img_col=DIMEN_X, img_depth=DIMEN_X, img_channel=1):
        if detectorNet is None:
            self.detectorNet = DetectorNet()
        else:
            self.detectorNet = detectorNet
        self.img_row = img_row
        self.img_col = img_col
        self.img_depth = img_depth
        self.img_channel = img_channel

    def get_classifier_net(self, X, coord, dnet_dropout_rate, cnet_dropout_rate):
        xsize = X.get_shape().as_list()
        # batch, depth, height, width, channels)
        X = tf.reshape(X, (-1, self.img_channel, self.img_depth, self.img_row, self.img_col))

        coordsize = coord.get_shape().as_list()

        coord = tf.reshape(coord, (-1, coordsize[2], coordsize[3], coordsize[4], coordsize[5]))
        _,_,_,_,_,_,_,_,noduleFeat, nodulePred = self.detectorNet.getDetectorNet(X, coord, dnet_dropout_rate)

        with tf.variable_scope('global/cl_scope'):
            nodulePred = tf.reshape(nodulePred, (-1, coordsize[1], coordsize[2]*coordsize[3]*coordsize[4]*coordsize[5]))

            featshape = noduleFeat.get_shape().as_list()
            # print(featshape)

            centerFeat = noduleFeat[:, :,
                         int(featshape[2] / 2 - 1):int(featshape[2] / 2 + 1),
                         int(featshape[3] / 2 - 1):int(featshape[3] / 2 + 1),
                         int(featshape[4] / 2 - 1):int(featshape[4] / 2 + 1)]
            # print(centerFeat.shape)
            centerFeat = tf.layers.max_pooling3d(centerFeat, pool_size=(1, 1, 1), strides=(1, 1, 1), padding="same",
                                                 data_format=self.DATA_FORMAT,
                                                 name="pool_cent_feat")
            # print(centerFeat.shape)

            centerFeat = centerFeat[:, :, 0, 0, 0]
            # print(centerFeat.shape)
            #out = tf.layers.dropout(centerFeat, rate=0.5, name="dropout_after_cent_feat")
            out = tf.layers.dropout(centerFeat, rate=cnet_dropout_rate, name="dropout_after_cent_feat")
            # print(out.shape)
            dense1 = tf.layers.dense(inputs=out, units=64, activation=tf.nn.relu, name="dense_1")
            # print(dense1.shape)
            dense2 = tf.layers.dense(inputs=dense1, units=1, activation=tf.nn.sigmoid, name="dense_2")
            out = tf.reshape(dense2, (-1, xsize[1]))
            #print(out.shape)
            baseline = tf.Variable(initial_value=-30.0, dtype=tf.float32, trainable=True)
            base_prob = tf.nn.sigmoid(baseline)
            # print(base_prob.shape)
            # print(base_prob)
            casePred = 1-tf.reduce_prod(1-out, axis=-1, keep_dims = True)*(1-base_prob)
            #print(casePred.shape)
            return nodulePred, casePred, out, centerFeat, noduleFeat


def get_model(trained_detector_net):
    net = ClassifierNet(trained_detector_net)
    return config, net


def get_config():
    return config

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, shape=(None, 5, 1, 96, 96, 96))
    coord = tf.placeholder(tf.float32, shape=(None, 5, 3, 24, 24, 24))

    net1 = DetectorNet()

    net2 = ClassifierNet(net1)

    net2.get_classifier_net(X, coord)