import tensorflow as tf
from Net.tensorflow_model.classifier_net import get_config, ClassiferNet
from Net.tensorflow_model.detector_net import DecetorNet

class ClassiferNetLoss(object):

    def __init__(self, config):
        self.config = config

    def getLoss(self, output, output_each, labels, isnod, batch_size, k_size):
        #loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[:, 0], logits=output))
        loss2 = tf.reduce_mean(-tf.reduce_sum(labels[:, 0] * tf.log(output), reduction_indices=[1]))
        missMask = tf.cast(output_each<self.config['miss_thresh'], dtype=tf.float32)
        missLoss = -tf.reduce_sum(missMask*isnod*tf.log(output_each+0.001))/batch_size/k_size
        return loss2+self.config['miss_ratio']*missLoss



if __name__ == "__main__":

    config = get_config()
    loss_object = ClassiferNetLoss(config=config)
    topK = config['topk']
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    isnod = tf.placeholder(tf.float32, shape=[None, topK])
    batch_size = 5
    X = tf.placeholder(tf.float32, shape=[None, topK, 1, 96, 96, 96])
    coord = tf.placeholder(tf.float32, shape=[None, topK, 3, 24, 24, 24])

    net1 = DecetorNet()

    net2 = ClassiferNet(net1)

    nodulePred, casePred, casePred_each = net2.getClassiferNet(X, coord)
    print(loss_object.getLoss(casePred, casePred_each, labels, isnod, batch_size, topK))


    with tf.Session() as sess:
        labels = tf.random_uniform(shape=[100,1])
        isnod = tf.random_uniform(shape=[100,topK])
        output = tf.random_uniform(shape=[100,1])
        output_each = tf.random_uniform(shape=[100,topK])

        sess.run(loss_object.getLoss(output, output_each, labels, isnod, batch_size, topK))
