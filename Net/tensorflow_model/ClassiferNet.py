import tensorflow as tf
from Net.tensorflow_model.DetectorNet import DecetorNet

class ClassiferNet(object):

    DATA_FORMAT = 'channels_first'

    def __init__(self, detectorNet, img_row=128, img_col=128, img_depth=128, img_channel=1):
        self.detectorNet = detectorNet
        self.img_row = img_row
        self.img_col = img_col
        self.img_depth = img_depth
        self.img_channel = img_channel

    def getClassiferNet(self, X, coord):
        xsize = X.get_shape().as_list()
        X = tf.reshape(X, (-1, self.img_channel, self.img_row, self.img_col, self.img_depth))
#        print(X.shape)
        coordsize = coord.get_shape().as_list()
        coord = tf.reshape(coord, (-1, coordsize[2], coordsize[3], coordsize[4], coordsize[5]))
        noduleFeat, nodulePred = self.detectorNet.getDetectorNet(X, coord)

        nodulePred = tf.reshape(nodulePred, (coordsize[0], coordsize[1], -1))

        featshape = noduleFeat.get_shape().as_list()
#        print(featshape)

        centerFeat = noduleFeat[:,:,
                     int(featshape[2] / 2 - 1):int(featshape[2] / 2 + 1),
                     int(featshape[3] / 2 - 1):int(featshape[3] / 2 + 1),
                     int(featshape[4] / 2 - 1):int(featshape[4] / 2 + 1)]
#        print(centerFeat.shape)
        centerFeat = tf.layers.max_pooling3d(centerFeat, pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                             data_format=self.DATA_FORMAT)
#        print(centerFeat.shape)

        centerFeat = centerFeat[:,:,0,0,0]
#        print(centerFeat.shape)
        out = tf.layers.dropout(centerFeat, rate=0.5)
#        print(out.shape)
        dense1 = tf.layers.dense(inputs=out, units=64, activation=tf.nn.relu)
#        print(dense1.shape)
        dense2 = tf.layers.dense(inputs=out, units=1, activation=tf.nn.sigmoid)
        out = tf.reshape(dense2, (xsize[0], xsize[1]))
#        print(out.shape)
#        print(self.baseline.shape)
        baseline = tf.constant(value=-30.0, dtype=tf.float32, shape=(xsize[0],))
        base_prob = tf.nn.sigmoid(baseline)
#        print(base_prob.shape)
#        print(base_prob)
        casePred = 1-tf.reduce_prod(1-out, axis=1)*(1-base_prob)
        return nodulePred, casePred, out



if __name__ == '__main__':
    X = tf.placeholder(tf.float32, shape=(100, 3, 1, 128, 128, 128))
    coord = tf.placeholder(tf.float32, shape=(100, 3, 3, 32, 32, 32))

    net1 = DecetorNet()

    net2 = ClassiferNet(net1)

    net2.getClassiferNet(X, coord)