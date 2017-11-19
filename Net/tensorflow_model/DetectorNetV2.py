import tensorflow as tf
from Net.Detector_Net_Loss_YOLO import DetectorNetLoss_YOLO
from Utils.utils import GetPBB
from Training.constants import DIMEN_X, DIMEN_Y
from collections import OrderedDict

config = {}
config['anchors'] = [10,30,60]
config['chanel'] = 1
config['crop_size'] = [DIMEN_X, DIMEN_X, DIMEN_X]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 6
config['sizelim2'] = 30
config['sizelim3'] = 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':False,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = []

class DecetorNetV2(object):

    DATA_FORMAT = 'channels_last'
    activation_op = "swish"

    def __init__(self, img_row=DIMEN_X, img_col=DIMEN_X, img_depth=DIMEN_X, img_channel=1):
        self.img_row = img_row
        self.img_col = img_col
        self.img_depth = img_depth
        self.img_channel = img_channel
        self.layers = OrderedDict()

    def fused_batch_normalization(self, input, name):

        # TODO: fused=True can improve this function performance but unfortunately TF can only support axis=3 if fused=True
        return tf.layers.batch_normalization(input, axis=4, momentum=0.1, epsilon=1e-05, fused=False, name=name)

    def build_resblock(self, input, cin, cout, name):
        conv1_name = name + "_conv1"
        res_conv1 = tf.layers.conv3d(input, cout, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                     padding="same", data_format=self.DATA_FORMAT, name=conv1_name)
        self.layers[conv1_name] = res_conv1

        res_conv1_bn = self.fused_batch_normalization(res_conv1, name=name+"_conv1_bn")
        res_conv1_relu = activation(res_conv1_bn, op=self.activation_op)

        conv2_name = name + "_conv2"
        res_conv2 = tf.layers.conv3d(res_conv1_relu, cout, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                     padding="same", data_format=self.DATA_FORMAT,
                                     name=conv2_name)
        self.layers[conv2_name] = res_conv2
        res_conv2_bn = self.fused_batch_normalization(res_conv2, name=name+"_conv2_bn")

        # RestNet shortcut
        res_shortcut = input
        if cin != cout:
            res_shortcut = tf.layers.conv3d(input, cout, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same",
                                            data_format=self.DATA_FORMAT,
                                            name=name+"_shortcut")
            self.layers[name+"_shortcut"] = res_shortcut
            res_shortcut = self.fused_batch_normalization(res_shortcut, name=name+"_shortcut_bn")

        res_op = tf.add(res_conv2_bn, res_shortcut, name=name+"_op")
        res_op_relu = activation(res_op, op=self.activation_op)

        return res_op_relu


    def getDetectorNet(self, X, coord):

        """make sure the input tensor has the expected shape"""
        if X.shape != (X.shape[0], self.img_channel, self.img_row, self.img_col, self.img_depth):
            raise ValueError("The input tensor does not have the expected shape. "
                             "The correct shape should be (batch_size, {}, {}, {}, {})"
                             .format(self.img_channel, self.img_row, self.img_col, self.img_depth))

        X = tf.transpose(X, perm=[0, 2, 3, 4, 1])
        coord = tf.transpose(coord, perm=[0, 2, 3, 4, 1])
        with tf.variable_scope('global/detector_scope'):
            # construct convBlock
            with tf.variable_scope('global/detector_scope/convBlock'):
                conv1 = tf.layers.conv3d(X, 16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                              data_format=self.DATA_FORMAT,
                                              name="conv1")
                self.layers["conv1"] = conv1
                conv1_nm = self.fused_batch_normalization(conv1, name="conv1_nm")
                conv1_relu = activation(conv1_nm, op=self.activation_op)

                conv2 = tf.layers.conv3d(conv1_relu, 16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                               padding="same", data_format=self.DATA_FORMAT, name="conv2")
                self.layers["conv2"] = conv2
                conv2_nm = self.fused_batch_normalization(conv2, name="conv2_nm")
                conv2_relu = activation(conv2_nm, op=self.activation_op)

                maxpool1 = tf.layers.max_pooling3d(conv2_relu, pool_size=(2, 2, 2), strides=(2, 2, 2),
                                                   padding="valid", data_format=self.DATA_FORMAT,
                                                   name="maxpool1")
            # construct resBlock1
            with tf.variable_scope('global/detector_scope/resBlock1'):
                res_block_1_1 = self.build_resblock(maxpool1, 16, 32, name="resBlock1-1")
                res_block_1_2 = self.build_resblock(res_block_1_1, 32, 32, name="resBlock1-2")
                maxpool2 = tf.layers.max_pooling3d(res_block_1_2, pool_size=(2, 2, 2), strides=(2, 2, 2),
                                                   padding="valid", data_format=self.DATA_FORMAT,
                                                   name="resblock1_maxpool")

            # construct resBlock2
            with tf.variable_scope("global/detector_scope/resBlock2"):
                res_block_2_1 = self.build_resblock(maxpool2, 32, 64, name="resBlock2-1")
                res_block_2_2 = self.build_resblock(res_block_2_1, 64, 64, name="resBlock2-2")
                maxpool3 = tf.layers.max_pooling3d(res_block_2_2, pool_size=(2, 2, 2), strides=(2, 2, 2),
                                                   padding="valid", data_format=self.DATA_FORMAT,
                                                   name="resblock2_maxpool")

            # construct resBlock3
            with tf.variable_scope("global/detector_scope/resBlock3"):
                resBlock3_1 = self.build_resblock(maxpool3, 64, 64, name="resBlock3-1")
                resBlock3_2 = self.build_resblock(resBlock3_1, 64, 64, name="resBlock3-2")
                resBlock3_3 = self.build_resblock(resBlock3_2, 64, 64, name="resBlock3-3")
                maxpool4 = tf.layers.max_pooling3d(resBlock3_3, pool_size=(2, 2, 2), strides=(2, 2, 2),
                                                   padding="valid",data_format=self.DATA_FORMAT,
                                                   name="resblock3_maxpool")
            # construct resBlock4
            with tf.variable_scope("global/detector_scope/resBlock4"):
                resBlock4_1 = self.build_resblock(maxpool4, 64, 64, name="resBlock4-1")
                resBlock4_2 = self.build_resblock(resBlock4_1, 64, 64, name="resBlock4-2")
                resBlock4_3 = self.build_resblock(resBlock4_2, 64, 64, name="resBlock4-3")

            with tf.variable_scope("global/detector_scope/up1"):
                up1 = tf.layers.conv3d_transpose(resBlock4_3, 64, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                                 use_bias=False, data_format=self.DATA_FORMAT,
                                                 name="up1")
                up1_bn = self.fused_batch_normalization(up1, name="up1_bn")
                up1_relu = activation(up1_bn, op=self.activation_op)
                comb3 = tf.concat([up1_relu, resBlock3_3], axis=4, name="comb")

            # construct resBlock5
            with tf.variable_scope("global/detector_scope/resBlock5"):
                resBlock5_1 = self.build_resblock(comb3, 128, 64, name="resBlock5_1")
                resBlock5_2 = self.build_resblock(resBlock5_1, 64, 64, name="resBlock5-2")
                resBlock5_3 = self.build_resblock(resBlock5_2, 64, 64, name="resBlock5-3")

            with tf.variable_scope("global/detector_scope/up2"):
                up2 = tf.layers.conv3d_transpose(resBlock5_3, 64, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                                     use_bias=False, data_format=self.DATA_FORMAT,
                                                     name="up2")
                up2_bn = self.fused_batch_normalization(up2, name="up2_bn")
                up2_relu = activation(up2_bn, op=self.activation_op)
                comb2 = tf.concat([up2_relu, res_block_2_2, coord], axis=4, name="comb")

            # construct resBlock6
            with tf.variable_scope("global/detector_scope/resBlock6"):
                resBlock6_1 = self.build_resblock(comb2, 131, 128, name="resBlock6_1")
                resBlock6_2 = self.build_resblock(resBlock6_1, 128, 128, name="resBlock6-2")
                feat = self.build_resblock(resBlock6_2, 128, 128, name="resBlock6-3")
                dropout = tf.layers.dropout(feat, rate=0.2, name="dropout_feature")

            with tf.variable_scope("global/detector_scope/output"):
                output_0 = tf.layers.conv3d(dropout, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                            data_format=self.DATA_FORMAT, name="output_conv1")
                self.layers["output_conv1"] = output_0
                output_relu = activation(output_0, op=self.activation_op)
                out = tf.layers.conv3d(output_relu, 5 * len(config['anchors']), kernel_size=(1, 1, 1),
                                       strides=(1, 1, 1), padding="valid", data_format=self.DATA_FORMAT,
                                       name="output")
                feat = tf.transpose(feat, perm=[0, 4, 1, 2, 3])

            return feat, out


# TODO: replace this with build-in leaky_relu after we upgrade to tensorflow version 1.4 or later
# This implementation uses less memory
def leaky_relu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def swish(x, name="swish"):
    with tf.variable_scope(name):
        return x*tf.nn.sigmoid(x)


def activation(x, leak=0.1, op="relu"):
    if op == "lrelu":
        return leaky_relu(x, leak=leak, name=op)
    elif op == "swish":
        return swish(x, name=op)
    return tf.nn.relu(x)


def get_model():
    net = DecetorNetV2()
    loss = DetectorNetLoss_YOLO()
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb


# explain for tf conv3d data shapes and see the descriptions at wiki below:
# https://github.com/skylimpid/AIHealth/wiki/TF-Links
def test_shapes():

    X = tf.random_uniform(shape=[100, 96, 96, 96, 1])
    print("X:", X.get_shape())
    conv1 = tf.layers.conv3d(X, 16, kernel_size=(20, 20, 20), strides=(2, 2, 2), padding="same")
    print("conv1:", conv1.get_shape())
    conv1_nm = tf.layers.batch_normalization(conv1, axis=4)
    print("conv1_nm:", conv1_nm.get_shape())
    conv1_relu = tf.nn.relu(conv1_nm)
    print("conv1_relu:", conv1_relu.get_shape())
    conv2 = tf.layers.conv3d(conv1_relu, 8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")
    print("conv2:",  conv2.get_shape())
    conv2_relu = tf.nn.relu(conv2)
    print("conv2_relu:",  conv2_relu.get_shape())

    maxpool = tf.layers.max_pooling3d(conv2_relu, pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same")
    print("maxpool:", maxpool.get_shape())

    logits = tf.layers.dense(inputs=maxpool, units=108)
    print("logits:", logits.get_shape())


if __name__ == '__main__':
    # if we specify a number instead of None, it works
    #X = tf.placeholder(tf.float32, shape=(100, 1, 128, 128, 128))
    X = tf.placeholder(tf.float32, shape=(None, 1, 96, 96, 96))

    # if we specify a number instead of None, it works
    #coord = tf.placeholder(tf.float32, shape=(100, 3, 32, 32, 32))
    coord = tf.placeholder(tf.float32, shape=(None, 3, 24, 24, 24))
    net = DecetorNetV2()
    net.getDetectorNet(X, coord)

    print (get_model())
    #test_shapes()

