import tensorflow as tf

from Training.Detector.TrainingDetectorData import TrainingDetectorData
from Net.tensorflow_model.DetectorNet import get_model
from Net.Loss import hard_mining

from Training.configuration_training import cfg

class DetectorTrainer(object):


    """
    Initializer

    """
    def __init__(self, config):

        # TO-DO: Initialize more parameters
        self.config = config
        datadir = cfg.DIR.preprocess_result_path
        # TO-DO: fill in the arguments in the initializer.
        self.build_model()
        self.dataset = TrainingDetectorData(datadir,
                                            '/home/xuan/lung_cancer_data/full.npy',
                                            self.net_config,
                                            phase='train')



    def has_positive_in_label(self, labels, train=True):

        labels = tf.reshape(tf.convert_to_tensor(labels), shape=(-1, 5))
        pos_idcs = labels[:, 0] > 0.5

        if tf.reduce_sum(tf.cast(pos_idcs, dtype=tf.int32)).eval() > 0:
            return True

        return False



    def train(self, sess):

        for epoch in range(0, self.config['num_of_epoch']):

            batch_count = 1

            while self.dataset.hasNextBatch():

                batch_data, batch_labels, batch_coord = self.dataset.getNextBatch(self.config['batch_size'])


                if (self.has_positive_in_label(batch_labels)):

                    sess.run([self.loss_1_optimizer],
                             feed_dict={self.X:batch_data, self.coord:batch_coord, self.labels:batch_labels})
                else:
                    sess.run([self.loss_2_optimizer],
                             feed_dict={self.X: batch_data, self.coord: batch_coord, self.labels: batch_labels})

                if batch_count % self.config['display_step']:
                    print("Current batch is %d" % batch_count)

                batch_count += 1

            print("Epoch %d finished." % epoch)
            self.dataset.reset()






    def build_model(self):

        self.X = tf.placeholder(tf.float32, shape=[None, 1, 128, 128, 128])
        self.coord = tf.placeholder(tf.float32, shape=[None, 3, 32, 32, 32])
        self.labels = tf.placeholder(tf.float32, shape=[None, 32, 32, 32, 3, 5])


        self.net_config, detector_net_object, loss_object, pbb = get_model()

        feat, out = detector_net_object.getDetectorNet(self.X, self.coord)

        [self.loss_1, self.loss_2] \
            = loss_object.getLoss(out, self.labels, train=True)


        self.loss_1_optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate']).minimize(self.loss_1)
        self.loss_2_optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate']).minimize(self.loss_2)



    def test(self):
        pass


    def validate(self):
        pass


if __name__ == "__main__":

    config = {'batch_size':1, 'learning_rate':0.01, 'num_of_epoch':10, 'display_step':2}

    instance = DetectorTrainer(config)
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        instance.train(sess)