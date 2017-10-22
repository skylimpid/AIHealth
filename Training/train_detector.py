import tensorflow as tf

from Training.Detector.TrainingDetectorData import TrainingDetectorData
from Net.tensorflow_model.DetectorNet import get_model
from Training.configuration_training import cfg
from Utils.split_combine import SplitComb
import os
import time
import numpy as np


class DetectorTrainer(object):


    """
    Initializer
    """
    def __init__(self, cfg):

        self.cfg = cfg

        self.build_model()

    def has_positive_in_label(self, labels):

        labels = tf.reshape(tf.convert_to_tensor(labels), shape=(-1, 5))
        pos_idcs = labels[:, 0] > 0.5

        if tf.reduce_sum(tf.cast(pos_idcs, dtype=tf.int32)).eval() > 0:
            return True

        return False

    def train(self, sess):
        value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope')
        saver = tf.train.Saver(value_list, max_to_keep=10)

        if not os.path.exists(cfg.DIR.detector_net_saver_dir):
            os.makedirs(cfg.DIR.detector_net_saver_dir)

        # Get the training data
        data_set = TrainingDetectorData(self.cfg.DIR.preprocess_result_path,
                                        self.cfg.DIR.detector_net_train_data_path,
                                        self.net_config,
                                        phase='train')

        start_time = time.time()

        for epoch in range(0, self.cfg.TRAIN.EPOCHS):

            batch_count = 1

            while data_set.hasNextBatch():

                batch_data, batch_labels, batch_coord = data_set.getNextBatch(self.cfg.TRAIN.BATCH_SIZE)

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
            data_set.reset()
            if epoch != 0 and epoch % self.cfg.TRAIN.SAVE_STEPS == 0:
                filename = self.cfg.DIR.detector_net_saver_file_prefix + '{:d}'.format(epoch+1)
                filename = os.path.join(self.cfg.DIR.detector_net_saver_dir, filename)
                saver.save(sess, filename, global_step=(epoch+1))

        filename = os.path.join(self.cfg.DIR.detector_net_saver_dir, (self.cfg.DIR.detector_net_saver_file_prefix
                                                                      + 'final'))
        saver.save(sess, filename)
        end_time = time.time()

        print("The total time used in training: {}".format(end_time-start_time))

    def build_model(self):

        self.X = tf.placeholder(tf.float32, shape=[None, 1, 128, 128, 128])
        self.coord = tf.placeholder(tf.float32, shape=[None, 3, 32, 32, 32])
        self.labels = tf.placeholder(tf.float32, shape=[None, 32, 32, 32, 3, 5])

        self.net_config, self.detector_net_object, loss_object, self.pbb = get_model()

        self.feat, self.out = self.detector_net_object.getDetectorNet(self.X, self.coord)

        [self.loss_1, self.loss_2] \
            = loss_object.getLoss(self.out, self.labels, train=True)

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

    def predict(self, sess):

        save_dir = os.path.join(self.cfg.DIR.bbox_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # load the previous trained detector_net model
        value_list = []
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope'))
        saver = tf.train.Saver(value_list, max_to_keep=100)
        saver.restore(sess, tf.train.latest_checkpoint(self.cfg.DIR.detector_net_saver_dir))

        # get input data
        margin = 32
        side_len = 64
        split_combine = SplitComb(side_len=side_len, max_stride=self.net_config['max_stride'],
                                  stride=self.net_config['stride'], margin=margin,
                                  pad_value=self.net_config['pad_value'])
        input_data = TrainingDetectorData(data_dir=self.cfg.DIR.preprocess_result_path,
                                          split_path=self.cfg.DIR.detector_net_train_data_path,
                                          config=self.net_config, split_comber=split_combine,
                                          phase='test')

        sess.run(tf.global_variables_initializer())
        start = time.time()
        for id in range(input_data.__len__()):
            imgs, bboxes, coord2, nzhw, filename = input_data.__getitem__(id)

            filename = filename.split('/')[-1].split('_')[0]
            print("Start to predict user:{}".format(filename))

            total_size_per_img = imgs.shape[0]

            index = 0
            final_out = None

            start_time = time.time()

            while index + self.cfg.TRAIN.BATCH_SIZE < total_size_per_img:
                feat_predict, out_predict = sess.run([self.feat, self.out], feed_dict={
                    self.X: imgs[index:index + self.cfg.TRAIN.BATCH_SIZE],
                    self.coord: coord2[index:index + self.cfg.TRAIN.BATCH_SIZE]})
                if final_out is None:
                    final_out = out_predict
                else:
                    final_out = tf.concat([final_out, out_predict], axis=0)

                index = index + self.cfg.TRAIN.BATCH_SIZE

            if index < total_size_per_img:
                feat_predict, out_predict = sess.run([self.feat, self.out], feed_dict={
                    self.X: imgs[index:], self.coord: coord2[index:]})
                if final_out is None:
                    final_out = out_predict
                else:
                    final_out = tf.concat([final_out, out_predict], axis=0)

            end_time = time.time()
            print("Predict user:{} spend: {}".format(filename, end_time - start_time))
            print("start to post-process the predict result for user:{}".format(filename))
            start_time = time.time()
            output = split_combine.combine(final_out, nzhw=nzhw)
            thresh = -3
            pbb, mask = self.pbb(output, thresh, ismask=True)
            np.save(os.path.join(save_dir, filename+'_pbb.npy'), pbb)
            np.save(os.path.join(save_dir, filename+'_lbb.npy'), bboxes)
            end_time = time.time()
            print("finish the post-process for user:{} spend:{}".format(filename, end_time - start_time))
        end = time.time()
        print("total process time:{}".format(end - start))


if __name__ == "__main__":

    instance = DetectorTrainer(cfg)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        instance.train(sess)
        #instance.predict(sess)