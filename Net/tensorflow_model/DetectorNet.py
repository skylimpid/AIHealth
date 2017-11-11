import tensorflow as tf
from Net.Detector_Net_Loss import DetectorNetLoss
from Utils.utils import GetPBB
from Training.constants import DIMEN_X, DIMEN_Y

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

class DecetorNet(object):

    DATA_FORMAT = 'channels_last'

    def __init__(self, img_row=DIMEN_X, img_col=DIMEN_X, img_depth=DIMEN_X, img_channel=1):
        self.img_row = img_row
        self.img_col = img_col
        self.img_depth = img_depth
        self.img_channel = img_channel

    def fused_batch_normalization(self, input_tensor, name):
        original_shape = input_tensor.get_shape().as_list()
        before_bn = tf.reshape(input_tensor, shape=(-1, original_shape[1], original_shape[2]* original_shape[3],
                                                    original_shape[4]))
        bn = tf.layers.batch_normalization(before_bn, axis=3, momentum=0.1, epsilon=1e-05, fused=True, name=name)
        after_bn = tf.reshape(bn, shape=(-1, original_shape[1], original_shape[2], original_shape[3],
                                         original_shape[4]))
        return after_bn

    def build_resblock(self, input, cin, cout, name):
        assert(((cout >= cin) and (cout-cin)%2 == 0) or (cin > cout))
        res_conv1 = tf.layers.conv3d(input, cout, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                     padding="same", data_format=self.DATA_FORMAT, name=name + "_conv1")
        res_conv1_bn = self.fused_batch_normalization(res_conv1, name=name+"_conv1_bn")
        res_conv1_relu = tf.nn.relu(res_conv1_bn, name=name+"_conv1_relu")
        res_conv2 = tf.layers.conv3d(res_conv1_relu, cout, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                     padding="same", data_format=self.DATA_FORMAT,
                                     name=name+"_conv2")
        res_conv2_bn = self.fused_batch_normalization(res_conv2, name=name+"_conv2_bn")
        # short cut
        if cin < cout:
            pad = int((cout-cin)/2)
            res_shortcut = tf.pad(input, tf.constant([[0, 0], [0, 0], [0, 0], [0, 0], [pad, pad]]),
                                  "CONSTANT", name=name+"_shortcut")
        elif cin == cout :
            res_shortcut = input
        else:
            res_shortcut = tf.layers.conv3d(input, cout, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same",
                                            data_format=self.DATA_FORMAT,
                                            name=name+"_shortcut")
            res_shortcut = self.fused_batch_normalization(res_shortcut, name=name+"_shortcut_bn")
        res_op = tf.add(res_conv2_bn, res_shortcut, name=name+"_op")
        res_op_relu = tf.nn.relu(res_op, name="res_op_relu")
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
                #conv1_nm = self.fused_batch_normalization(conv1, name="conv1_nm")
                conv1_nm = tf.layers.batch_normalization(conv1, axis=4)
                conv1_relu = tf.nn.relu(conv1_nm, name="conv1_relu")

                conv2 = tf.layers.conv3d(conv1_relu, 16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                               padding="same", data_format=self.DATA_FORMAT, name="conv2")
                conv2_nm = self.fused_batch_normalization(conv2, name="conv2_nm")
                conv2_relu = tf.nn.relu(conv2_nm, name="conv2_relu")

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
                up1_relu = tf.nn.relu(up1_bn, name="up1_relu")
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
                up2_relu = tf.nn.relu(up2_bn, name="up2_relu")
                comb2 = tf.concat([up2_relu, res_block_2_2, coord], axis=4, name="comb")

            # construct resBlock6
            with tf.variable_scope("global/detector_scope/resBlock6"):
                resBlock6_1 = self.build_resblock(comb2, 131, 128, name="resBlock6_1")
                resBlock6_2 = self.build_resblock(resBlock6_1, 128, 128, name="resBlock6-2")
                feat = self.build_resblock(resBlock6_2, 128, 128, name="resBlock6-3")
                dropout = tf.layers.dropout(feat, rate=0.2, name="dropout_feature")

            with tf.variable_scope("global/detector_scope/output"):
                output_0 = tf.layers.conv3d(dropout, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                            data_format=self.DATA_FORMAT)
                output_relu = tf.nn.relu(output_0)
                output_2 = tf.layers.conv3d(output_relu, 5 * len(config['anchors']), kernel_size=(1, 1, 1),
                                            strides=(1, 1, 1), padding="valid", data_format=self.DATA_FORMAT)

                feat = tf.transpose(feat, perm=[0, 4, 1, 2, 3])
                output_2 = tf.transpose(output_2, perm=[0, 4, 1, 2, 3])
                size = output_2.get_shape().as_list()
                out = tf.reshape(output_2, (-1, size[1], size[2] * size[3] * size[4]))
                out = tf.transpose(out, perm=(0, 2, 1))
                out = tf.reshape(out, (-1, size[2], size[3], size[4], len(config['anchors']), 5))
                #print(out.shape)
            return conv1, res_block_1_2, res_block_2_2, resBlock3_3, resBlock4_3, comb3, resBlock5_3, comb2, feat, out


def get_model():
    net = DecetorNet()
    loss = DetectorNetLoss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb


if __name__ == '__main__':
    # if we specify a number instead of None, it works
    #X = tf.placeholder(tf.float32, shape=(100, 1, 128, 128, 128))
    X = tf.placeholder(tf.float32, shape=(None, 1, 96, 96, 96))

    # if we specify a number instead of None, it works
    #coord = tf.placeholder(tf.float32, shape=(100, 3, 32, 32, 32))
    coord = tf.placeholder(tf.float32, shape=(None, 3, 24, 24, 24))
    net = DecetorNet()
    net.getDetectorNet(X, coord)

    print (get_model())