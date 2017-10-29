import tensorflow as tf
import os
import time
import numpy as np
import shutil

from Training.Detector.TrainingDetectorData import TrainingDetectorData
from Net.tensorflow_model.DetectorNet import get_model
from Training.configuration_training import cfg
from Utils.split_combine import SplitComb
from Training.constants import DETECTOR_NET_TENSORBOARD_LOG_DIR
from Training.constants import DIMEN_X, DIMEN_Y


class DetectorTrainer(object):


    """
    Initializer
    """
    def __init__(self, cfg):

        self.cfg = cfg

        self.build_model()


    # Detect if the provided tensor 'labels' contains +1 labels.
    def has_positive_in_label(self, labels):

        labels = labels.reshape((-1, 5))
        pos_idcs = labels[:, 0] > 0.5

        if np.sum(pos_idcs) > 0:
            return True

        return False

    def has_negative_in_label(self, labels):
        labels = labels.reshape((-1, 5))
        neg_idcs = labels[:, 0] < -0.5

        if np.sum(neg_idcs) > 0:
            return True

        return False

    def need_hard_mining(self, labels, hard_minng):
        labels = labels.reshape((-1, 5))
        pos_idcs = labels[:, 0] < -0.5

        if np.sum(pos_idcs) > hard_minng:
            return True

        return False

    def train(self, sess, clear=True):
        if clear:
            if os.path.exists(DETECTOR_NET_TENSORBOARD_LOG_DIR):
                shutil.rmtree(DETECTOR_NET_TENSORBOARD_LOG_DIR)

        loss_pos_neg_holder = tf.placeholder(tf.float32)
        loss_pos_neg_tensor = tf.summary.scalar("loss_pos_neg", loss_pos_neg_holder)
        loss_pos_holder = tf.placeholder(tf.float32)
        loss_pos_tensor = tf.summary.scalar("loss_pos", loss_pos_holder)
        loss_neg_holder = tf.placeholder(tf.float32)
        loss_neg_tensor = tf.summary.scalar("loss_neg", loss_neg_holder)
        writer = tf.summary.FileWriter(DETECTOR_NET_TENSORBOARD_LOG_DIR)
        writer.add_graph(sess.graph)

        saver = tf.train.Saver(max_to_keep=10)

        if not os.path.exists(cfg.DIR.detector_net_saver_dir):
            os.makedirs(cfg.DIR.detector_net_saver_dir)

        # Get the training data
        data_set = TrainingDetectorData(self.cfg.DIR.preprocess_result_path,
                                        self.cfg.DIR.detector_net_train_data_path,
                                        self.net_config,
                                        phase='train')

        start_time = time.time()
        index = 1
        loss_pos_neg = 0
        previous_loss_pos_neg = 0
        loss_pos = 0
        previous_loss_pos = 0
        loss_neg = 0
        previous_loss_neg = 0
        tf.get_default_graph().finalize()
        for epoch in range(1, self.cfg.TRAIN.EPOCHS+1):

            batch_count = 0
            batch_step = 0
            while data_set.hasNextBatch():
                use_previous_loss = False
                batch_data, batch_labels, batch_coord = data_set.getNextBatch(self.cfg.TRAIN.BATCH_SIZE)
                if self.has_positive_in_label(batch_labels):
                    if self.has_negative_in_label(batch_labels):
                        if self.need_hard_mining(batch_labels, self.cfg.TRAIN.BATCH_SIZE * self.net_config['num_hard']):
                            _, loss_pos_neg = sess.run([self.classify_loss_with_pos_neg_with_hard_mining_optimizer,
                                                        self.classify_loss_with_pos_neg_with_hard_mining],
                                                       feed_dict={self.X: batch_data, self.coord: batch_coord,
                                                                  self.labels: batch_labels})
                        else:
                            _, loss_pos_neg = sess.run([self.classify_loss_with_pos_neg_without_hard_mining_optimizer,
                                                        self.classify_loss_with_pos_neg_without_hard_mining],
                                                       feed_dict={self.X: batch_data, self.coord: batch_coord,
                                                                  self.labels: batch_labels})
                    else:
                        _, loss_pos = sess.run([self.classify_loss_without_neg_optimizer, self.classify_loss_without_neg],
                                               feed_dict={self.X: batch_data, self.coord: batch_coord,
                                                          self.labels: batch_labels})
                else:
                    if self.has_negative_in_label(batch_labels):
                        if self.need_hard_mining(batch_labels, self.cfg.TRAIN.BATCH_SIZE * self.net_config['num_hard']):
                            _, loss_neg = sess.run([self.classify_loss_without_pos_with_hard_mining_optimizer,
                                                    self.classify_loss_without_pos_with_hard_mining],
                                                   feed_dict={self.X: batch_data, self.coord: batch_coord,
                                                              self.labels: batch_labels})
                        else:
                            _, loss_neg = sess.run([self.classify_loss_without_pos_without_hard_mining_optimizer,
                                                    self.classify_loss_without_pos_without_hard_mining],
                                                   feed_dict={self.X: batch_data, self.coord: batch_coord,
                                                              self.labels: batch_labels})
                    else:
                        print("Can not find any label data from the data-set in this batch. Skip it")
                        use_previous_loss = True

                if use_previous_loss:
                    loss_pos_neg = previous_loss_pos_neg
                    loss_pos = previous_loss_pos
                    loss_neg = previous_loss_neg

                feed = {loss_pos_neg_holder: loss_pos_neg, loss_pos_holder: loss_pos, loss_neg_holder: loss_neg}
                loss_pos_neg_str, loss_pos_str, loss_neg_str = sess.run([loss_pos_neg_tensor, loss_pos_tensor,
                                                                         loss_neg_tensor], feed_dict=feed)
                writer.add_summary(loss_pos_neg_str, index)
                writer.add_summary(loss_pos_str, index)
                writer.add_summary(loss_neg_str, index)
                index += 1
                writer.flush()
                previous_loss_pos_neg = loss_pos_neg
                previous_loss_pos = loss_pos
                previous_loss_neg = loss_neg

                batch_step += 1
                if batch_step % self.cfg.TRAIN.DISPLAY_STEPS == 0:
                    print("Batching step: %d" % batch_step)

                batch_count += len(batch_labels)

            print("Epoch %d finished on %d batches." % (epoch, batch_count))

            data_set.reset()
            if epoch != 0 and epoch % self.cfg.TRAIN.SAVE_STEPS == 0:
                filename = self.cfg.DIR.detector_net_saver_file_prefix + '{:d}'.format(epoch+1)
                filename = os.path.join(self.cfg.DIR.detector_net_saver_dir, filename)
                saver.save(sess, filename, global_step=(epoch+1))

        filename = os.path.join(self.cfg.DIR.detector_net_saver_dir, (self.cfg.DIR.detector_net_saver_file_prefix
                                                                      + 'final'))


        """
        TO-DO: we need to set up period check point save, and support resuming training from checkpoint
        """
        saver.save(sess, filename)
        end_time = time.time()

        print("The total time used in training: {}".format(end_time-start_time))


    def build_model(self):

        self.X = tf.placeholder(tf.float32, shape=[None, 1, DIMEN_X, DIMEN_X, DIMEN_X])
        self.coord = tf.placeholder(tf.float32, shape=[None, 3, DIMEN_Y, DIMEN_Y, DIMEN_Y])
        self.labels = tf.placeholder(tf.float32, shape=[None, DIMEN_Y, DIMEN_Y, DIMEN_Y, 3, 5])

        self.net_config, self.detector_net_object, loss_object, self.pbb = get_model()

        self.feat, self.out = self.detector_net_object.getDetectorNet(self.X, self.coord)

        [self.classify_loss_with_pos_neg_without_hard_mining,
         self.classify_loss_without_pos_without_hard_mining,
         self.classify_loss_without_neg,
         self.classify_loss_with_pos_neg_with_hard_mining,
         self.classify_loss_without_pos_with_hard_mining] = loss_object.getLoss(self.out, self.labels,
                                                                                self.cfg.TRAIN.BATCH_SIZE)

        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(self.cfg.TRAIN.LEARNING_RATE, global_step,
                                        self.cfg.TRAIN.LEARNING_RATE_STEP_SIZE, 0.1, staircase=True)

        self.classify_loss_with_pos_neg_without_hard_mining_optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.classify_loss_with_pos_neg_without_hard_mining, global_step=global_step)

        self.classify_loss_without_pos_without_hard_mining_optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.classify_loss_without_pos_without_hard_mining, global_step=global_step)

        self.classify_loss_without_neg_optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.classify_loss_without_neg, global_step=global_step)

        self.classify_loss_with_pos_neg_with_hard_mining_optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.classify_loss_with_pos_neg_with_hard_mining, global_step=global_step)

        self.classify_loss_without_pos_with_hard_mining_optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.classify_loss_without_pos_with_hard_mining, global_step=global_step)

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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        instance.train(sess)
        #instance.predict(sess)