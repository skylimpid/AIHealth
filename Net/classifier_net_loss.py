import tensorflow as tf
from Net.tensorflow_model.classifier_net import get_config, ClassifierNet
from Net.tensorflow_model.detector_net import DetectorNet

class ClassifierNetLoss(object):

    def __init__(self, config):
        self.config = config

    def getLoss(self, output, output_each, labels, isnod, batch_size, k_size):
        #loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output))
        output = tf.clip_by_value(output, clip_value_min=1e-5, clip_value_max=(1-(1e-5)))
        loss2 = -tf.reduce_sum(labels * tf.log(output) + (1-labels) * tf.log(1 - output))
        missMask = tf.cast(output_each<self.config['miss_thresh'], dtype=tf.float32)
        missLoss = -tf.reduce_sum(missMask*isnod*tf.log(output_each+1e-3))/batch_size/k_size
        #missLoss = -tf.reduce_sum(tf.multiply(tf.multiply(missMask, isnod), tf.log(output_each + 0.001))) / batch_size / k_size
        return tf.add(loss2, self.config['miss_ratio']*missLoss), loss2, missLoss, missMask


if __name__ == "__main__":

    config = get_config()
    loss_object = ClassifierNetLoss(config=config)
    topK = config['topk']
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    isnod = tf.placeholder(tf.float32, shape=[None, topK])
    batch_size = 5
    X = tf.placeholder(tf.float32, shape=[None, topK, 1, 96, 96, 96])
    coord = tf.placeholder(tf.float32, shape=[None, topK, 3, 24, 24, 24])

    detector_net = DetectorNet()
    cl_net = ClassifierNet(detector_net)
    nodulePred, casePred, casePred_each, centerFeat, noduleFeat = cl_net.get_classifier_net(X, coord)
    print(loss_object.getLoss(casePred, casePred_each, labels, isnod, batch_size, topK))

    with tf.Session() as sess:
        labels = tf.random_uniform(shape=[100,1])
        isnod = tf.random_uniform(shape=[100,topK])
        output = tf.random_uniform(shape=[100,1])
        output_each = tf.random_uniform(shape=[100,topK])

        loss = sess.run(loss_object.getLoss(output, output_each, labels, isnod, batch_size, topK))
        print("loss: ", loss)
