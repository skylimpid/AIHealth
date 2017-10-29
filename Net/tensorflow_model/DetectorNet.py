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


    def getDetectorNet(self, X, coord):

        """make sure the input tensor has the expected shape"""
        if X.shape != (X.shape[0], self.img_channel, self.img_row, self.img_col, self.img_depth):
            raise ValueError("The input tensor does not have the expected shape. "
                             "The correct shape should be (batch_size, {}, {}, {}, {})"
                             .format(self.img_channel, self.img_row, self.img_col, self.img_depth))

        X = tf.transpose(X, perm=[0, 2, 3, 4, 1])
        coord = tf.transpose(coord, perm=[0, 2, 3, 4, 1])
        with tf.variable_scope('global/detector_scope'):
            # construct preblock
            preBlock_0 = tf.layers.conv3d(X, 5, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                          data_format=self.DATA_FORMAT,
                                          name="pre_block_conv3d_1_after_input")
            # preBlock_1 = tf.layers.batch_normalization(preBlock_0, axis=1, momentum=0.1, epsilon=1e-05)
            preBlock_1 = self.fused_batch_normalization(preBlock_0, name="bn_pre_block_conv3d_1")
            preBlock_relu1 = tf.nn.relu(preBlock_1, name="relu_pre_block_conv3d_1")

            preBlock_3 = tf.layers.conv3d(preBlock_relu1, 5, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                          data_format=self.DATA_FORMAT, name="pre_block_conv3d_2")
            # preBlock_4 = tf.layers.batch_normalization(preBlock_3, axis=1, momentum=0.1, epsilon=1e-05)
            preBlock_4 = self.fused_batch_normalization(preBlock_3, name="bn_pre_block_conv3d_2")
            preBlock_relu2 = tf.nn.relu(preBlock_4, name="relu_pre_block_conv3d_2")
            # print(preBlock_relu2.shape)

            maxpool1 = tf.layers.max_pooling3d(preBlock_relu2, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                               data_format=self.DATA_FORMAT, name="maxpool_pre_block")
            #print(maxpool1.shape)

            # the first forward
            forw1_0_conv1 = tf.layers.conv3d(maxpool1, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT,
                                             name="forw1_conv3d_1_after_pre_block")
            # forw1_0_bn1 = tf.layers.batch_normalization(forw1_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw1_0_bn1 = self.fused_batch_normalization(forw1_0_conv1, name="bn_forw1_conv3d_1")
            forw1_0_relu1 = tf.nn.relu(forw1_0_bn1, name="relu_forw1_conv3d_1")
            forw1_0_conv2 = tf.layers.conv3d(forw1_0_relu1, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                             padding="same", data_format=self.DATA_FORMAT,
                                             name="forw1_conv3d_2")
            #forw1_0_bn2 = tf.layers.batch_normalization(forw1_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw1_0_bn2 = self.fused_batch_normalization(forw1_0_conv2, name="bn_forw1_conv3d_2")
            #print(forw1_0_bn2.shape)
            # forward1 short cut
            forw1_0_shortcut_0 = tf.layers.conv3d(maxpool1, 32, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                                  padding="valid", data_format=self.DATA_FORMAT,
                                                  name="forw1_short_cut_after_maxpool_pre_block")

            # forw1_0_shortcut_1 = tf.layers.batch_normalization(forw1_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
            forw1_0_shortcut_1 = self.fused_batch_normalization(forw1_0_shortcut_0, name="bn_forw1_short_cut")
            #print(forw1_0_shortcut_1.shape)
            forw1_0_added = tf.add(forw1_0_bn2, forw1_0_shortcut_1, name="bn_forw1_conv3d_2_ADD_bn_forw1_short_cut")
            forw1_0_relu = tf.nn.relu(forw1_0_added, name="forw1_add_0_relu")
            #print(forw1_0_relu.shape)

            forw1_1_conv1 = tf.layers.conv3d(forw1_0_relu, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw1_conv3d_3")
            # forw1_1_bn1 = tf.layers.batch_normalization(forw1_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw1_1_bn1 = self.fused_batch_normalization(forw1_1_conv1, name="bn_forw1_conv3d_3")
            forw1_1_relu1 = tf.nn.relu(forw1_1_bn1, name="relu_forw1_conv3d_3")
            forw1_1_conv2 = tf.layers.conv3d(forw1_1_relu1, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                             padding="same", data_format=self.DATA_FORMAT,
                                             name="forw1_conv3d_4")
            # forw1_1_bn2 = tf.layers.batch_normalization(forw1_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw1_1_bn2 = self.fused_batch_normalization(forw1_1_conv2, name="bn_forw1_conv3d_4")
            #print(forw1_1_bn2.shape)
            forw1_1_added = tf.add(forw1_1_bn2, forw1_0_relu, name="bn_forw1_conv3d_4_ADD_forw1_add_0_relu")
            forw1_1_relu = tf.nn.relu(forw1_1_added, name="forw1_add_1_relu")
            #print(forw1_1_relu.shape)
            maxpool2 = tf.layers.max_pooling3d(forw1_1_relu, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                               data_format=self.DATA_FORMAT, name="forw1_maxpool")
            #print(maxpool2.shape)

            # the second forward
            forw2_0_conv1 = tf.layers.conv3d(maxpool2, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT,
                                             name="forw2_conv3d_1_after_forw1_maxpool")
            # forw2_0_bn1 = tf.layers.batch_normalization(forw2_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw2_0_bn1 = self.fused_batch_normalization(forw2_0_conv1, name="bn_forw2_conv3d_1")
            forw2_0_relu1 = tf.nn.relu(forw2_0_bn1, name="relu_forw2_conv3d_1")
            forw2_0_conv2 = tf.layers.conv3d(forw2_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                             padding="same", data_format=self.DATA_FORMAT,
                                             name="forw2_conv3d_2")
            # forw2_0_bn2 = tf.layers.batch_normalization(forw2_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw2_0_bn2 = self.fused_batch_normalization(forw2_0_conv2, name="bn_forw2_conv3d_2")
            # forward2 short cut
            forw2_0_shortcut_0 = tf.layers.conv3d(maxpool2, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                                  padding="valid", data_format=self.DATA_FORMAT,
                                                  name="forw2_short_cut_after_maxpool_forw1")
            # forw2_0_shortcut_1 = tf.layers.batch_normalization(forw2_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
            forw2_0_shortcut_1 = self.fused_batch_normalization(forw2_0_shortcut_0, name="bn_forw2_short_cut")
            forw2_0_added = tf.add(forw2_0_bn2, forw2_0_shortcut_1, name="bn_forw2_conv3d_2_ADD_bn_forw2_short_cut")
            forw2_0_relu = tf.nn.relu(forw2_0_added, name="forw2_add_0_relu")

            forw2_1_conv1 = tf.layers.conv3d(forw2_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw2_conv3d_3")
            # forw2_1_bn1 = tf.layers.batch_normalization(forw2_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw2_1_bn1 = self.fused_batch_normalization(forw2_1_conv1, name="bn_forw2_conv3d_3")
            forw2_1_relu1 = tf.nn.relu(forw2_1_bn1, name="relu_forw2_conv3d_3")
            forw2_1_conv2 = tf.layers.conv3d(forw2_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                             padding="same", data_format=self.DATA_FORMAT,
                                             name="forw2_conv3d_4")
            # forw2_1_bn2 = tf.layers.batch_normalization(forw2_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw2_1_bn2 = self.fused_batch_normalization(forw2_1_conv2, name="bn_forw2_conv3d_4")
            forw2_1_added = tf.add(forw2_1_bn2, forw2_0_relu, name="bn_forw2_conv3d_4_ADD_forw2_add_0_relu")
            forw2_1_relu = tf.nn.relu(forw2_1_added, name="forw2_add_1_relu")

            maxpool3 = tf.layers.max_pooling3d(forw2_1_relu, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                               data_format=self.DATA_FORMAT, name="forw2_maxpool")
            #print (maxpool3.shape)

            # the third forward
            forw3_0_conv1 = tf.layers.conv3d(maxpool3, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT,
                                             name="forw3_conv3d_1_after_forw2_maxpool")
            # forw3_0_bn1 = tf.layers.batch_normalization(forw3_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw3_0_bn1 = self.fused_batch_normalization(forw3_0_conv1, name="bn_forw3_conv3d_1")
            forw3_0_relu1 = tf.nn.relu(forw3_0_bn1, name="relu_forw3_conv3d_1")
            forw3_0_conv2 = tf.layers.conv3d(forw3_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                             padding="same", data_format=self.DATA_FORMAT,
                                             name="forw3_conv3d_2")
            # forw3_0_bn2 = tf.layers.batch_normalization(forw3_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw3_0_bn2 = self.fused_batch_normalization(forw3_0_conv2, name="bn_forw3_conv3d_2")
            # forward3 short cut
            forw3_0_shortcut_0 = tf.layers.conv3d(maxpool3, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                                  padding="valid", data_format=self.DATA_FORMAT,
                                                  name="forw3_short_cut_after_maxpool_forw2")
            # forw3_0_shortcut_1 = tf.layers.batch_normalization(forw3_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
            forw3_0_shortcut_1 = self.fused_batch_normalization(forw3_0_shortcut_0, name="bn_forw3_short_cut")
            forw3_0_added = tf.add(forw3_0_bn2, forw3_0_shortcut_1, name="bn_forw3_conv3d_2_ADD_bn_forw3_short_cut")
            forw3_0_relu = tf.nn.relu(forw3_0_added, name="forw3_add_0_relu")

            forw3_1_conv1 = tf.layers.conv3d(forw3_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw3_conv3d_3")
            # forw3_1_bn1 = tf.layers.batch_normalization(forw3_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw3_1_bn1 = self.fused_batch_normalization(forw3_1_conv1, name="bn_forw3_conv3d_3")
            forw3_1_relu1 = tf.nn.relu(forw3_1_bn1, name="relu_forw3_conv3d_3")
            forw3_1_conv2 = tf.layers.conv3d(forw3_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw3_conv3d_4")
            # forw3_1_bn2 = tf.layers.batch_normalization(forw3_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw3_1_bn2 = self.fused_batch_normalization(forw3_1_conv2, name="bn_forw3_conv3d_4")
            forw3_1_added = tf.add(forw3_1_bn2, forw3_0_relu, name="bn_forw3_conv3d_4_ADD_forw3_add_0_relu")
            forw3_1_relu = tf.nn.relu(forw3_1_added, name="forw3_add_1_relu")

            forw3_2_conv1 = tf.layers.conv3d(forw3_1_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw3_conv3d_5")
            # forw3_2_bn1 = tf.layers.batch_normalization(forw3_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw3_2_bn1 = self.fused_batch_normalization(forw3_2_conv1, name="bn_forw3_conv3d_5")
            forw3_2_relu1 = tf.nn.relu(forw3_2_bn1, name="relu_forw3_conv3d_5")
            forw3_2_conv2 = tf.layers.conv3d(forw3_2_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                             padding="same", data_format=self.DATA_FORMAT,
                                             name="forw3_conv3d_6")
            # forw3_2_bn2 = tf.layers.batch_normalization(forw3_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw3_2_bn2 = self.fused_batch_normalization(forw3_2_conv2, name="bn_forw3_conv3d_6")
            forw3_2_added = tf.add(forw3_2_bn2, forw3_1_relu, name="bn_forw3_conv3d_6_ADD_forw3_add_1_relu")
            forw3_2_relu = tf.nn.relu(forw3_2_added, name="forw3_add_2_relu")

            maxpool4 = tf.layers.max_pooling3d(forw3_2_relu, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                               data_format=self.DATA_FORMAT, name="forw3_maxpool")
            #print (maxpool4.shape)

            # the fourth forward
            forw4_0_conv1 = tf.layers.conv3d(maxpool4, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT,
                                             name="forw4_conv3d_1_after_forw3_maxpool")
            # forw4_0_bn1 = tf.layers.batch_normalization(forw4_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw4_0_bn1 = self.fused_batch_normalization(forw4_0_conv1, name="bn_forw4_conv3d_1")
            forw4_0_relu1 = tf.nn.relu(forw4_0_bn1, name="relu_forw4_conv3d_1")
            forw4_0_conv2 = tf.layers.conv3d(forw4_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw4_conv3d_2")
            # forw4_0_bn2 = tf.layers.batch_normalization(forw4_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw4_0_bn2 = self.fused_batch_normalization(forw4_0_conv2, name="bn_forw4_conv3d_2")
            # forward4 short cut
            forw4_0_shortcut_0 = tf.layers.conv3d(maxpool4, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                                  data_format=self.DATA_FORMAT,
                                                  name="forw4_short_cut_after_maxpool_forw3")
            # forw4_0_shortcut_1 = tf.layers.batch_normalization(forw4_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
            forw4_0_shortcut_1 = self.fused_batch_normalization(forw4_0_shortcut_0, name="bn_forw4_short_cut")
            forw4_0_added = tf.add(forw4_0_bn2, forw4_0_shortcut_1, name="bn_forw4_conv3d_2_ADD_bn_forw4_short_cut")
            forw4_0_relu = tf.nn.relu(forw4_0_added, name="forw4_add_0_relu")

            forw4_1_conv1 = tf.layers.conv3d(forw4_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw4_conv3d_3")
            # forw4_1_bn1 = tf.layers.batch_normalization(forw4_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw4_1_bn1 = self.fused_batch_normalization(forw4_1_conv1, name="bn_forw4_conv3d_3")
            forw4_1_relu1 = tf.nn.relu(forw4_1_bn1, name="relu_forw4_conv3d_3")
            forw4_1_conv2 = tf.layers.conv3d(forw4_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw4_conv3d_4")
            # forw4_1_bn2 = tf.layers.batch_normalization(forw4_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw4_1_bn2 = self.fused_batch_normalization(forw4_1_conv2, name="bn_forw4_conv3d_4")
            forw4_1_added = tf.add(forw4_1_bn2, forw4_0_relu, name="bn_forw4_conv3d_4_ADD_forw4_add_0_relu")
            forw4_1_relu = tf.nn.relu(forw4_1_added, name="forw4_add_1_relu")

            forw4_2_conv1 = tf.layers.conv3d(forw4_1_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw4_conv3d_5")
            # forw4_2_bn1 = tf.layers.batch_normalization(forw4_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            forw4_2_bn1 = self.fused_batch_normalization(forw4_2_conv1, name="bn_forw4_conv3d_5")
            forw4_2_relu1 = tf.nn.relu(forw4_2_bn1, name="relu_forw4_conv3d_5")
            forw4_2_conv2 = tf.layers.conv3d(forw4_2_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="forw4_conv3d_6")
            # forw4_2_bn2 = tf.layers.batch_normalization(forw4_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            forw4_2_bn2 = self.fused_batch_normalization(forw4_2_conv2, name="bn_forw4_conv3d_6")
            forw4_2_added = tf.add(forw4_2_bn2, forw4_1_relu, name="bn_forw4_conv3d_6_ADD_forw4_add_1_relu")
            forw4_2_relu = tf.nn.relu(forw4_2_added, name="forw4_add_2_relu")
            #print (forw4_2_relu.shape)

            # Path 1
            path1_0 = tf.layers.conv3d_transpose(forw4_2_relu, 64, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                                 use_bias=False, data_format=self.DATA_FORMAT,
                                                 name="path_1_conv3d_transpose_after_forw4")
            # path1_1 = tf.layers.batch_normalization(path1_0, axis=1, momentum=0.1, epsilon=1e-05)
            path1_1 = self.fused_batch_normalization(path1_0, name="bn_path_1")
            path1_0_relu1 = tf.nn.relu(path1_1, name="relu_path_1")

            # comb3
            comb3 = tf.concat([path1_0_relu1, forw3_2_relu], axis=4, name="path_1_concat_forw3")
            #print (comb3.shape)

            # back3
            back3_0_conv1 = tf.layers.conv3d(comb3, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT,
                                             name="back3_conv3d_1_after_path_1_concat_forw3")
            # back3_0_bn1 = tf.layers.batch_normalization(back3_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            back3_0_bn1 = self.fused_batch_normalization(back3_0_conv1, name="bn_back3_conv3d_1")
            back3_0_relu1 = tf.nn.relu(back3_0_bn1, name="relu_back3_conv3d_1")
            back3_0_conv2 = tf.layers.conv3d(back3_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back3_conv3d_2")
            # back3_0_bn2 = tf.layers.batch_normalization(back3_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            back3_0_bn2 = self.fused_batch_normalization(back3_0_conv2, name="bn_back3_conv3d_2")
            # back3 short cut
            back3_0_shortcut_0 = tf.layers.conv3d(comb3, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                                  data_format=self.DATA_FORMAT,
                                                  name="back3_short_cut_after_path_1_concat_forw3")
            # back3_0_shortcut_1 = tf.layers.batch_normalization(back3_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
            back3_0_shortcut_1 = self.fused_batch_normalization(back3_0_shortcut_0, name="bn_back3_short_cut")
            back3_0_added = tf.add(back3_0_bn2, back3_0_shortcut_1, name="bn_back3_conv3d_2_ADD_bn_back3_short_cut")
            back3_0_relu = tf.nn.relu(back3_0_added, name="back3_add_0_relu")

            back3_1_conv1 = tf.layers.conv3d(back3_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back3_conv3d_3")
            # back3_1_bn1 = tf.layers.batch_normalization(back3_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            back3_1_bn1 = self.fused_batch_normalization(back3_1_conv1, name="bn_back3_conv3d_3")
            back3_1_relu1 = tf.nn.relu(back3_1_bn1, name="relu_back3_conv3d_3")
            back3_1_conv2 = tf.layers.conv3d(back3_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back3_conv3d_4")
            # back3_1_bn2 = tf.layers.batch_normalization(back3_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            back3_1_bn2 = self.fused_batch_normalization(back3_1_conv2, name="bn_back3_conv3d_4")
            back3_1_added = tf.add(back3_1_bn2, back3_0_relu, name="bn_back3_conv3d_4_ADD_back3_add_0_relu")
            back3_1_relu = tf.nn.relu(back3_1_added, name="back3_add_1_relu")

            back3_2_conv1 = tf.layers.conv3d(back3_1_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back3_conv3d_5")
            # back3_2_bn1 = tf.layers.batch_normalization(back3_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            back3_2_bn1 = self.fused_batch_normalization(back3_2_conv1, name="bn_back3_conv3d_5")
            back3_2_relu1 = tf.nn.relu(back3_2_bn1, name="relu_back3_conv3d_5")
            back3_2_conv2 = tf.layers.conv3d(back3_2_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back3_conv3d_6")
            # back3_2_bn2 = tf.layers.batch_normalization(back3_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            back3_2_bn2 = self.fused_batch_normalization(back3_2_conv2, name="bn_back3_conv3d_6")
            back3_2_added = tf.add(back3_2_bn2, back3_1_relu, name="bn_back3_conv3d_6_ADD_back3_add_1_relu")
            back3_2_relu = tf.nn.relu(back3_2_added, name="back3_add_2_relu")
            #print (back3_2_relu.shape)

            # Path 2
            path2_0 = tf.layers.conv3d_transpose(back3_2_relu, 64, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                                 use_bias=False, data_format=self.DATA_FORMAT,
                                                 name="path_2_after_back3")
            # path2_1 = tf.layers.batch_normalization(path2_0, axis=1, momentum=0.1, epsilon=1e-05)
            path2_1 = self.fused_batch_normalization(path2_0, name="bn_path_2")
            path2_0_relu1 = tf.nn.relu(path2_1, name="relu_path_2")
            #print(path2_0_relu1.shape)

            # comb2
            # print(path2_0_relu1.shape)
            comb2 = tf.concat([path2_0_relu1, forw2_1_relu, coord], axis=4, name="path_2_concat_forw2")
            # print(comb2.shape)

            # back 2
            back2_0_conv1 = tf.layers.conv3d(comb2, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT,
                                             name="back2_conv3d_1_after_path_2_concat_forw2")
            # back2_0_bn1 = tf.layers.batch_normalization(back2_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            back2_0_bn1 = self.fused_batch_normalization(back2_0_conv1, name="bn_back2_conv3d_1")
            back2_0_relu1 = tf.nn.relu(back2_0_bn1, name="relu_back2_conv3d_1")
            back2_0_conv2 = tf.layers.conv3d(back2_0_relu1, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back2_conv3d_2")
            # back2_0_bn2 = tf.layers.batch_normalization(back2_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            back2_0_bn2 = self.fused_batch_normalization(back2_0_conv2, name="bn_back2_conv3d_2")
            # back2 short cut
            back2_0_shortcut_0 = tf.layers.conv3d(comb2, 128, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                                  data_format=self.DATA_FORMAT,
                                                  name="back2_short_cut_after_path_2_concat_forw2")
            # back2_0_shortcut_1 = tf.layers.batch_normalization(back2_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
            back2_0_shortcut_1 = self.fused_batch_normalization(back2_0_shortcut_0, name="bn_back2_short_cut")
            back2_0_added = tf.add(back2_0_bn2, back2_0_shortcut_1, name="bn_back2_conv3d_2_ADD_bn_back2_short_cut")
            back2_0_relu = tf.nn.relu(back2_0_added, name="back2_add_0_relu")

            back2_1_conv1 = tf.layers.conv3d(back2_0_relu, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back2_conv3d_3")
            # back2_1_bn1 = tf.layers.batch_normalization(back2_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            back2_1_bn1 = self.fused_batch_normalization(back2_1_conv1, name="bn_back2_conv3d_3")
            back2_1_relu1 = tf.nn.relu(back2_1_bn1, name="relu_back2_conv3d_3")
            back2_1_conv2 = tf.layers.conv3d(back2_1_relu1, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back2_conv3d_4")
            # back2_1_bn2 = tf.layers.batch_normalization(back2_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            back2_1_bn2 = self.fused_batch_normalization(back2_1_conv2, name="bn_back2_conv3d_4")
            back2_1_added = tf.add(back2_1_bn2, back2_0_relu, name="bn_back2_conv3d_4_ADD_back2_add_0_relu")
            back2_1_relu = tf.nn.relu(back2_1_added, name="back2_add_1_relu")

            back2_2_conv1 = tf.layers.conv3d(back2_1_relu, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back2_conv3d_5")
            # back2_2_bn1 = tf.layers.batch_normalization(back2_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
            back2_2_bn1 = self.fused_batch_normalization(back2_2_conv1, name="bn_back2_conv3d_5")
            back2_2_relu1 = tf.nn.relu(back2_2_bn1, name="relu_back2_conv3d_5")
            back2_2_conv2 = tf.layers.conv3d(back2_2_relu1, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                             data_format=self.DATA_FORMAT, name="back2_conv3d_6")
            # back2_2_bn2 = tf.layers.batch_normalization(back2_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
            back2_2_bn2 = self.fused_batch_normalization(back2_2_conv2, name="bn_back2_conv3d_6")
            back2_2_added = tf.add(back2_2_bn2, back2_1_relu, name="bn_back2_conv3d_6_ADD_back2_add_1_relu")

            feat = tf.nn.relu(back2_2_added, name="relu_back_2")
            dropout_2 = tf.layers.dropout(feat, rate=0.2, name="dropout_feature")
            # print(dropout_2.shape)

            # Output
            output_0 = tf.layers.conv3d(dropout_2, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                        data_format=self.DATA_FORMAT)
            output_relu = tf.nn.relu(output_0)
            output_2 = tf.layers.conv3d(output_relu, 5 * len(config['anchors']), kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                        padding="valid", data_format=self.DATA_FORMAT)

            feat = tf.transpose(feat, perm=[0, 4, 1, 2, 3])
            output_2 = tf.transpose(output_2, perm=[0, 4, 1, 2, 3])
            # print(feat.shape)
            # print(output_2.shape)

            size = output_2.get_shape().as_list()
            # print(size)
            out = tf.reshape(output_2, (-1, size[1], size[2] * size[3] * size[4]))
            out = tf.transpose(out, perm=(0, 2, 1))
            out = tf.reshape(out, (-1, size[2], size[3], size[4], len(config['anchors']), 5))
            # print (out.shape)
            # print (feat.shape)

            return feat, out

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