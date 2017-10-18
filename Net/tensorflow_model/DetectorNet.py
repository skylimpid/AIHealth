import tensorflow as tf
from Net.Detector_Net_Loss import DetectorNetLoss
from Utils.utils import GetPBB


config = {}
config['anchors'] = [10,30,60]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
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
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = []

class DecetorNet(object):

    DATA_FORMAT = 'channels_first'

    def __init__(self, img_row=128, img_col=128, img_depth=128, img_channel=1):
        self.img_row = img_row
        self.img_col = img_col
        self.img_depth = img_depth
        self.img_channel = img_channel

    def fused_batch_normalization(self, input_tensor):
        original_shape = input_tensor.get_shape().as_list()
        before_bn = tf.reshape(input_tensor, shape=(-1, original_shape[1], original_shape[2],
                                                    original_shape[3]*original_shape[4]))
        bn = tf.layers.batch_normalization(before_bn, axis=1, momentum=0.1, epsilon=1e-05, fused=True)
        after_bn = tf.reshape(bn, shape=(-1, original_shape[1], original_shape[2], original_shape[3],
                                         original_shape[4]))
        return after_bn


    def getDetectorNet(self, X, coord):

        """make sure the input tensor has the expected shape"""
        if X.shape != (X.shape[0], self.img_channel, self.img_row, self.img_col, self.img_depth):
            raise ValueError("The input tensor does not have the expected shape. "
                             "The correct shape should be (batch_size, {}, {}, {}, {})"
                             .format(self.img_channel, self.img_row, self.img_col, self.img_depth))

#        print(type(X))
        # construct preblock
        preBlock_0 = tf.layers.conv3d(X, 24, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", use_bias=False,
                                      data_format=self.DATA_FORMAT)
        # preBlock_1 = tf.layers.batch_normalization(preBlock_0, axis=1, momentum=0.1, epsilon=1e-05)
        preBlock_1 = self.fused_batch_normalization(preBlock_0)
        preBlock_relu1 = tf.nn.relu(preBlock_1)
        preBlock_3 = tf.layers.conv3d(preBlock_relu1, 24, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                      use_bias=False, data_format=self.DATA_FORMAT)
        # preBlock_4 = tf.layers.batch_normalization(preBlock_3, axis=1, momentum=0.1, epsilon=1e-05)
        preBlock_4 = self.fused_batch_normalization(preBlock_3)
        preBlock_relu2 = tf.nn.relu(preBlock_4)
        # print(preBlock_relu2.shape)

        maxpool1 = tf.layers.max_pooling3d(preBlock_relu2, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                           data_format=self.DATA_FORMAT)
        #print(maxpool1.shape)

        # the first forward
        forw1_0_conv1 = tf.layers.conv3d(maxpool1, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw1_0_bn1 = tf.layers.batch_normalization(forw1_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw1_0_bn1 = self.fused_batch_normalization(forw1_0_conv1)
        forw1_0_relu1 = tf.nn.relu(forw1_0_bn1)
        forw1_0_conv2 = tf.layers.conv3d(forw1_0_relu1, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        #forw1_0_bn2 = tf.layers.batch_normalization(forw1_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw1_0_bn2 = self.fused_batch_normalization(forw1_0_conv2)
        #print(forw1_0_bn2.shape)
        # forward1 short cut
        forw1_0_shortcut_0 = tf.layers.conv3d(maxpool1, 32, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                              use_bias=False, data_format=self.DATA_FORMAT)

        # forw1_0_shortcut_1 = tf.layers.batch_normalization(forw1_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
        forw1_0_shortcut_1 = self.fused_batch_normalization(forw1_0_shortcut_0)
        #print(forw1_0_shortcut_1.shape)
        forw1_0_added = tf.add(forw1_0_bn2, forw1_0_shortcut_1)
        forw1_0_relu = tf.nn.relu(forw1_0_added)
        #print(forw1_0_relu.shape)

        forw1_1_conv1 = tf.layers.conv3d(forw1_0_relu, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw1_1_bn1 = tf.layers.batch_normalization(forw1_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw1_1_bn1 = self.fused_batch_normalization(forw1_1_conv1)
        forw1_1_relu1 = tf.nn.relu(forw1_1_bn1)
        forw1_1_conv2 = tf.layers.conv3d(forw1_1_relu1, 32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw1_1_bn2 = tf.layers.batch_normalization(forw1_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw1_1_bn2 = self.fused_batch_normalization(forw1_1_conv2)
        #print(forw1_1_bn2.shape)
        forw1_1_added = tf.add(forw1_1_bn2, forw1_0_relu)
        forw1_1_relu = tf.nn.relu(forw1_1_added)
        #print(forw1_1_relu.shape)
        maxpool2 = tf.layers.max_pooling3d(forw1_1_relu, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                           data_format=self.DATA_FORMAT)
        #print(maxpool2.shape)

        # the second forward
        forw2_0_conv1 = tf.layers.conv3d(maxpool2, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw2_0_bn1 = tf.layers.batch_normalization(forw2_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw2_0_bn1 = self.fused_batch_normalization(forw2_0_conv1)
        forw2_0_relu1 = tf.nn.relu(forw2_0_bn1)
        forw2_0_conv2 = tf.layers.conv3d(forw2_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw2_0_bn2 = tf.layers.batch_normalization(forw2_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw2_0_bn2 = self.fused_batch_normalization(forw2_0_conv2)
        # forward2 short cut
        forw2_0_shortcut_0 = tf.layers.conv3d(maxpool2, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                              use_bias=False, data_format=self.DATA_FORMAT)
        # forw2_0_shortcut_1 = tf.layers.batch_normalization(forw2_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
        forw2_0_shortcut_1 = self.fused_batch_normalization(forw2_0_shortcut_0)
        forw2_0_added = tf.add(forw2_0_bn2, forw2_0_shortcut_1)
        forw2_0_relu = tf.nn.relu(forw2_0_added)

        forw2_1_conv1 = tf.layers.conv3d(forw2_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw2_1_bn1 = tf.layers.batch_normalization(forw2_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw2_1_bn1 = self.fused_batch_normalization(forw2_1_conv1)
        forw2_1_relu1 = tf.nn.relu(forw2_1_bn1)
        forw2_1_conv2 = tf.layers.conv3d(forw2_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw2_1_bn2 = tf.layers.batch_normalization(forw2_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw2_1_bn2 = self.fused_batch_normalization(forw2_1_conv2)
        forw2_1_added = tf.add(forw2_1_bn2, forw2_0_relu)
        forw2_1_relu = tf.nn.relu(forw2_1_added)

        maxpool3 = tf.layers.max_pooling3d(forw2_1_relu, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                           data_format=self.DATA_FORMAT)
        #print (maxpool3.shape)

        # the third forward
        forw3_0_conv1 = tf.layers.conv3d(maxpool3, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw3_0_bn1 = tf.layers.batch_normalization(forw3_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw3_0_bn1 = self.fused_batch_normalization(forw3_0_conv1)
        forw3_0_relu1 = tf.nn.relu(forw3_0_bn1)
        forw3_0_conv2 = tf.layers.conv3d(forw3_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw3_0_bn2 = tf.layers.batch_normalization(forw3_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw3_0_bn2 = self.fused_batch_normalization(forw3_0_conv2)
        # forward3 short cut
        forw3_0_shortcut_0 = tf.layers.conv3d(maxpool3, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                              use_bias=False, data_format=self.DATA_FORMAT)
        # forw3_0_shortcut_1 = tf.layers.batch_normalization(forw3_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
        forw3_0_shortcut_1 = self.fused_batch_normalization(forw3_0_shortcut_0)
        forw3_0_added = tf.add(forw3_0_bn2, forw3_0_shortcut_1)
        forw3_0_relu = tf.nn.relu(forw3_0_added)

        forw3_1_conv1 = tf.layers.conv3d(forw3_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw3_1_bn1 = tf.layers.batch_normalization(forw3_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw3_1_bn1 = self.fused_batch_normalization(forw3_1_conv1)
        forw3_1_relu1 = tf.nn.relu(forw3_1_bn1)
        forw3_1_conv2 = tf.layers.conv3d(forw3_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw3_1_bn2 = tf.layers.batch_normalization(forw3_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw3_1_bn2 = self.fused_batch_normalization(forw3_1_conv2)
        forw3_1_added = tf.add(forw3_1_bn2, forw3_0_relu)
        forw3_1_relu = tf.nn.relu(forw3_1_added)

        forw3_2_conv1 = tf.layers.conv3d(forw3_1_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw3_2_bn1 = tf.layers.batch_normalization(forw3_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw3_2_bn1 = self.fused_batch_normalization(forw3_2_conv1)
        forw3_2_relu1 = tf.nn.relu(forw3_2_bn1)
        forw3_2_conv2 = tf.layers.conv3d(forw3_2_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw3_2_bn2 = tf.layers.batch_normalization(forw3_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw3_2_bn2 = self.fused_batch_normalization(forw3_2_conv2)
        forw3_2_added = tf.add(forw3_2_bn2, forw3_1_relu)
        forw3_2_relu = tf.nn.relu(forw3_2_added)

        maxpool4 = tf.layers.max_pooling3d(forw3_2_relu, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                           data_format=self.DATA_FORMAT)
        #print (maxpool4.shape)

        # the fourth forward
        forw4_0_conv1 = tf.layers.conv3d(maxpool4, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw4_0_bn1 = tf.layers.batch_normalization(forw4_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw4_0_bn1 = self.fused_batch_normalization(forw4_0_conv1)
        forw4_0_relu1 = tf.nn.relu(forw4_0_bn1)
        forw4_0_conv2 = tf.layers.conv3d(forw4_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw4_0_bn2 = tf.layers.batch_normalization(forw4_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw4_0_bn2 = self.fused_batch_normalization(forw4_0_conv2)
        # forward4 short cut
        forw4_0_shortcut_0 = tf.layers.conv3d(maxpool4, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                              use_bias=False, data_format=self.DATA_FORMAT)
        # forw4_0_shortcut_1 = tf.layers.batch_normalization(forw4_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
        forw4_0_shortcut_1 = self.fused_batch_normalization(forw4_0_shortcut_0)
        forw4_0_added = tf.add(forw4_0_bn2, forw4_0_shortcut_1)
        forw4_0_relu = tf.nn.relu(forw4_0_added)

        forw4_1_conv1 = tf.layers.conv3d(forw4_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw4_1_bn1 = tf.layers.batch_normalization(forw4_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw4_1_bn1 = self.fused_batch_normalization(forw4_1_conv1)
        forw4_1_relu1 = tf.nn.relu(forw4_1_bn1)
        forw4_1_conv2 = tf.layers.conv3d(forw4_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw4_1_bn2 = tf.layers.batch_normalization(forw4_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw4_1_bn2 = self.fused_batch_normalization(forw4_1_conv2)
        forw4_1_added = tf.add(forw4_1_bn2, forw4_0_relu)
        forw4_1_relu = tf.nn.relu(forw4_1_added)

        forw4_2_conv1 = tf.layers.conv3d(forw4_1_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw4_2_bn1 = tf.layers.batch_normalization(forw4_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        forw4_2_bn1 = self.fused_batch_normalization(forw4_2_conv1)
        forw4_2_relu1 = tf.nn.relu(forw4_2_bn1)
        forw4_2_conv2 = tf.layers.conv3d(forw4_2_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # forw4_2_bn2 = tf.layers.batch_normalization(forw4_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        forw4_2_bn2 = self.fused_batch_normalization(forw4_2_conv2)
        forw4_2_added = tf.add(forw4_2_bn2, forw4_1_relu)
        forw4_2_relu = tf.nn.relu(forw4_2_added)
        #print (forw4_2_relu.shape)

        # Path 1
        path1_0 = tf.layers.conv3d_transpose(forw4_2_relu, 64, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                             use_bias=False, data_format=self.DATA_FORMAT)
        # path1_1 = tf.layers.batch_normalization(path1_0, axis=1, momentum=0.1, epsilon=1e-05)
        path1_1 = self.fused_batch_normalization(path1_0)
        path1_0_relu1 = tf.nn.relu(path1_1)

        # comb3
        comb3 = tf.concat([path1_0_relu1, forw3_2_relu], axis=1)
        #print (comb3.shape)

        # back3
        back3_0_conv1 = tf.layers.conv3d(comb3, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back3_0_bn1 = tf.layers.batch_normalization(back3_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        back3_0_bn1 = self.fused_batch_normalization(back3_0_conv1)
        back3_0_relu1 = tf.nn.relu(back3_0_bn1)
        back3_0_conv2 = tf.layers.conv3d(back3_0_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back3_0_bn2 = tf.layers.batch_normalization(back3_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        back3_0_bn2 = self.fused_batch_normalization(back3_0_conv2)
        # back3 short cut
        back3_0_shortcut_0 = tf.layers.conv3d(comb3, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                              use_bias=False, data_format=self.DATA_FORMAT)
        # back3_0_shortcut_1 = tf.layers.batch_normalization(back3_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
        back3_0_shortcut_1 = self.fused_batch_normalization(back3_0_shortcut_0)
        back3_0_added = tf.add(back3_0_bn2, back3_0_shortcut_1)
        back3_0_relu = tf.nn.relu(back3_0_added)

        back3_1_conv1 = tf.layers.conv3d(back3_0_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back3_1_bn1 = tf.layers.batch_normalization(back3_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        back3_1_bn1 = self.fused_batch_normalization(back3_1_conv1)
        back3_1_relu1 = tf.nn.relu(back3_1_bn1)
        back3_1_conv2 = tf.layers.conv3d(back3_1_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back3_1_bn2 = tf.layers.batch_normalization(back3_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        back3_1_bn2 = self.fused_batch_normalization(back3_1_conv2)
        back3_1_added = tf.add(back3_1_bn2, back3_0_relu)
        back3_1_relu = tf.nn.relu(back3_1_added)

        back3_2_conv1 = tf.layers.conv3d(back3_1_relu, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back3_2_bn1 = tf.layers.batch_normalization(back3_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        back3_2_bn1 = self.fused_batch_normalization(back3_2_conv1)
        back3_2_relu1 = tf.nn.relu(back3_2_bn1)
        back3_2_conv2 = tf.layers.conv3d(back3_2_relu1, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back3_2_bn2 = tf.layers.batch_normalization(back3_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        back3_2_bn2 = self.fused_batch_normalization(back3_2_conv2)
        back3_2_added = tf.add(back3_2_bn2, back3_1_relu)
        back3_2_relu = tf.nn.relu(back3_2_added)
        #print (back3_2_relu.shape)

        # Path 2
        path2_0 = tf.layers.conv3d_transpose(back3_2_relu, 64, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                             use_bias=False, data_format=self.DATA_FORMAT)
        # path2_1 = tf.layers.batch_normalization(path2_0, axis=1, momentum=0.1, epsilon=1e-05)
        path2_1 = self.fused_batch_normalization(path2_0)
        path2_0_relu1 = tf.nn.relu(path2_1)
        #print(path2_0_relu1.shape)

        # comb2
#        print(path2_0_relu1.shape)
        comb2 = tf.concat([path2_0_relu1, forw2_1_relu, coord], axis=1)
#        print(comb2.shape)

        # back 2
        back2_0_conv1 = tf.layers.conv3d(comb2, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back2_0_bn1 = tf.layers.batch_normalization(back2_0_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        back2_0_bn1 = self.fused_batch_normalization(back2_0_conv1)
        back2_0_relu1 = tf.nn.relu(back2_0_bn1)
        back2_0_conv2 = tf.layers.conv3d(back2_0_relu1, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back2_0_bn2 = tf.layers.batch_normalization(back2_0_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        back2_0_bn2 = self.fused_batch_normalization(back2_0_conv2)
        # back2 short cut
        back2_0_shortcut_0 = tf.layers.conv3d(comb2, 128, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                              use_bias=False, data_format=self.DATA_FORMAT)
        # back2_0_shortcut_1 = tf.layers.batch_normalization(back2_0_shortcut_0, axis=1, momentum=0.1, epsilon=1e-05)
        back2_0_shortcut_1 = self.fused_batch_normalization(back2_0_shortcut_0)
        back2_0_added = tf.add(back2_0_bn2, back2_0_shortcut_1)
        back2_0_relu = tf.nn.relu(back2_0_added)

        back2_1_conv1 = tf.layers.conv3d(back2_0_relu, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back2_1_bn1 = tf.layers.batch_normalization(back2_1_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        back2_1_bn1 = self.fused_batch_normalization(back2_1_conv1)
        back2_1_relu1 = tf.nn.relu(back2_1_bn1)
        back2_1_conv2 = tf.layers.conv3d(back2_1_relu1, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back2_1_bn2 = tf.layers.batch_normalization(back2_1_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        back2_1_bn2 = self.fused_batch_normalization(back2_1_conv2)
        back2_1_added = tf.add(back2_1_bn2, back2_0_relu)
        back2_1_relu = tf.nn.relu(back2_1_added)

        back2_2_conv1 = tf.layers.conv3d(back2_1_relu, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back2_2_bn1 = tf.layers.batch_normalization(back2_2_conv1, axis=1, momentum=0.1, epsilon=1e-05)
        back2_2_bn1 = self.fused_batch_normalization(back2_2_conv1)
        back2_2_relu1 = tf.nn.relu(back2_2_bn1)
        back2_2_conv2 = tf.layers.conv3d(back2_2_relu1, 128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                         use_bias=False, data_format=self.DATA_FORMAT)
        # back2_2_bn2 = tf.layers.batch_normalization(back2_2_conv2, axis=1, momentum=0.1, epsilon=1e-05)
        back2_2_bn2 = self.fused_batch_normalization(back2_2_conv2)
        back2_2_added = tf.add(back2_2_bn2, back2_1_relu)

        feat = tf.nn.relu(back2_2_added)
        dropout_2 = tf.layers.dropout(feat, rate=0.2)
#        print(dropout_2.shape)

        # Output
        output_0 = tf.layers.conv3d(dropout_2, 64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                    use_bias=False, data_format=self.DATA_FORMAT)
        output_relu = tf.nn.relu(output_0)
        output_2 = tf.layers.conv3d(output_relu, 5 * len(config['anchors']), kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                    padding="valid", use_bias=False, data_format=self.DATA_FORMAT)
#        print(feat.shape)
#        print(output_2.shape)

        size = output_2.get_shape().as_list()
#        print(size)
        out = tf.reshape(output_2, (-1, size[1], size[2] * size[3] * size[4]))
        out = tf.transpose(out, perm=(0, 2, 1))
        out = tf.reshape(out, (-1, size[2], size[3], size[4], len(config['anchors']), 5))
#        print (out.shape)
        return feat, out

def get_model():
    net = DecetorNet()
    loss = DetectorNetLoss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb


if __name__ == '__main__':
    # if we specify a number instead of None, it works
    #X = tf.placeholder(tf.float32, shape=(100, 1, 128, 128, 128))
    X = tf.placeholder(tf.float32, shape=(None, 1, 128, 128, 128))

    # if we specify a number instead of None, it works
    #coord = tf.placeholder(tf.float32, shape=(100, 3, 32, 32, 32))
    coord = tf.placeholder(tf.float32, shape=(None, 3, 32, 32, 32))
    net = DecetorNet()
    net.getDetectorNet(X, coord)

    print (get_model())