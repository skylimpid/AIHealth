import tensorflow as tf


class ClassiferNetLoss(object):

    def __init__(self, config):
        self.config = config

    def getLoss(self, output, output_each, labels, isnod, batch_size, k_size):
        loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[:, 0], logits=output))
        missMask = tf.cast(output_each<self.config['miss_thresh'], dtype=tf.float32)
        missLoss = -tf.reduce_sum(missMask*isnod*tf.log(output_each+0.001))/batch_size/k_size
        return loss2+self.config['miss_ratio']*missLoss