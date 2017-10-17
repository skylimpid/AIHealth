import tensorflow as tf

from Training.Detector.TrainingDetectorData import TrainingDetectorData
from Net.tensorflow_model.DetectorNet import get_model
from Training.configuration_training import cfg
import os
import time


class DetectorTrainer(object):


    """
    Initializer
    """
    def __init__(self, cfg):

        self.cfg = cfg
        if not os.path.exists(cfg.DIR.detector_net_saver_dir):
            os.makedirs(cfg.DIR.detector_net_saver_dir)

        self.build_model()
        self.data_set = TrainingDetectorData(self.cfg.DIR.preprocess_result_path,
                                             self.cfg.DIR.detector_net_train_data_path,
                                             self.net_config,
                                             phase='train')

    def has_positive_in_label(self, labels):

        labels = tf.reshape(tf.convert_to_tensor(labels), shape=(-1, 5))
        pos_idcs = labels[:, 0] > 0.5

        if tf.reduce_sum(tf.cast(pos_idcs, dtype=tf.int32)).eval() > 0:
            return True

        return False

    def train(self, sess):

        saver = tf.train.Saver(max_to_keep=10)

        start_time = time.time()

        for epoch in range(0, self.cfg.TRAIN.EPOCHS):

            batch_count = 1

            while self.data_set.hasNextBatch():

                batch_data, batch_labels, batch_coord = self.data_set.getNextBatch(self.cfg.TRAIN.BATCH_SIZE)

                if self.has_positive_in_label(batch_labels):
                    sess.run([self.loss_1_optimizer],
                             feed_dict={self.X:batch_data, self.coord:batch_coord, self.labels:batch_labels})
                else:
                    sess.run([self.loss_2_optimizer],
                             feed_dict={self.X: batch_data, self.coord: batch_coord, self.labels: batch_labels})

                if batch_count % self.cfg.TRAIN.DISPLAY_STEPS:
                    print("Current batch is %d" % batch_count)

                batch_count += 1

            print("Epoch %d finished." % epoch)
            self.data_set.reset()
            if epoch != 0 and (epoch % cfg.TRAIN.SAVE_STEPS == 0 or epoch == (self.cfg.TRAIN.EPOCHS - 1)):
                filename = (cfg.DIR.detector_net_saver_file_prefix + '_iter_{:d}'.format(epoch+1) + '.ckpt')
                filename = os.path.join(cfg.DIR.detector_net_saver_dir, filename)
                saver.save(sess, filename, global_step=(epoch+1))

        end_time = time.time()

        print("The total time used in training: {}".format(end_time-start_time))

    def build_model(self):

        self.X = tf.placeholder(tf.float32, shape=[None, 1, 128, 128, 128])
        self.coord = tf.placeholder(tf.float32, shape=[None, 3, 32, 32, 32])
        self.labels = tf.placeholder(tf.float32, shape=[None, 32, 32, 32, 3, 5])

        self.net_config, detector_net_object, loss_object, pbb = get_model()

        feat, out = detector_net_object.getDetectorNet(self.X, self.coord)

        [self.loss_1, self.loss_2] \
            = loss_object.getLoss(out, self.labels, train=True)

        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(self.cfg.TRAIN.LEARNING_RATE, global_step,
                                        self.cfg.TRAIN.LEARNING_RATE_STEP_SIZE, 0.1, staircase=True)

        self.loss_1_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.loss_1, global_step=global_step)

        self.loss_2_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.loss_2, global_step=global_step)

    def test(self):
        pass

    def validate(self):
        pass


if __name__ == "__main__":

    instance = DetectorTrainer(cfg)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        instance.train(sess)