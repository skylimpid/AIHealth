import os
import time
import shutil
import numpy as np
import tensorflow as tf

from Training.Detector.training_detector_data import TrainingDetectorData
from Net.tensorflow_model.detector_net_2 import get_model
from Training.configuration_training import cfg
from Utils.split_combine import SplitComb
from Training.constants import DIMEN_X, DIMEN_Y, MARGIN, SIDE_LEN
from Training.configuration_training import DETECTOR_NET_TENSORBOARD_LOG_DIR
from Utils.nms_cython import nms, iou

#######
# Note: not tested yet
#######
class DetectorTrainer(object):

    """
    Initializer
    """
    def __init__(self, cfg):

        self.cfg = cfg

        self.build_model()

        self.validate_average_iou_holder = tf.placeholder(tf.float32)
        self.validate_average_iou_tensor = tf.summary.scalar("val_avg_iou", self.validate_average_iou_holder)

        self.validate_nodule_predict_ratio_holder = tf.placeholder(tf.float32)
        self.validate_nodule_predict_ratio_tensor = tf.summary.scalar("val_avg_nodule_predict_ratio",
                                                                      self.validate_nodule_predict_ratio_holder)

    # Detect if the provided tensor 'labels' contains +1 labels.
    def num_pos_in_label(self, labels):

        labels = labels.reshape((-1, 5))
        pos_idcs = labels[:, 0] > 0.5

        return np.sum(pos_idcs).astype(np.float32)


    def train(self, sess, continue_training=False, clear=True, enable_validate=False):
        if clear:
            if os.path.exists(DETECTOR_NET_TENSORBOARD_LOG_DIR):
                shutil.rmtree(DETECTOR_NET_TENSORBOARD_LOG_DIR)

        # initialize the global parameters
        sess.run(tf.global_variables_initializer())

        # run in tf debug mode. Needs to launch the program on commandLine
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # load previous saved weights if we enable the continue_training
        if continue_training:
            value_list = []
            value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope'))
            restore = tf.train.Saver(value_list)
            restore.restore(sess, tf.train.latest_checkpoint(self.cfg.DIR.detector_net_saver_dir))

        average_loss_holder = tf.placeholder(tf.float32)
        average_loss_tensor = tf.summary.scalar("avg_loss", average_loss_holder)

        cur_loss_holder = tf.placeholder(tf.float32)
        cur_loss_tensor = tf.summary.scalar("cur_loss", cur_loss_holder)

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
        tf.get_default_graph().finalize()
        train_step = 0
        for epoch in range(1, self.cfg.TRAIN.EPOCHS+1):

            batch_count = 0
            batch_step = 0
            total_loss = 0
            latest_total_loss = 0
            latest_avg_loss = 0
            latest_count = 0
            while data_set.hasNextBatch():
                train_step += 1
                loss_per_batch = 0
                batch_data, batch_labels, batch_coord = data_set.getNextBatch(self.cfg.TRAIN.BATCH_SIZE)
                condition = self.num_pos_in_label(batch_labels)

                _, loss_per_batch, summary_op_batch = sess.run([self.optimizer, self.loss, self.summary_op],
                                                              feed_dict={self.X: batch_data, self.coord: batch_coord,
                                                                         self.labels: batch_labels,
                                                                         self.condition: condition})
                writer.add_summary(summary_op_batch, train_step)

                latest_total_loss += loss_per_batch
                latest_count += 1
                total_loss += loss_per_batch
                batch_step += 1
                if batch_step % self.cfg.TRAIN.DISPLAY_STEPS == 0:
                    latest_avg_loss = latest_total_loss / latest_count
                    print("Step: %d, Avg Loss: %f, Cur Loss: %f" % (batch_step, (total_loss/batch_step), latest_avg_loss))
                    latest_total_loss = 0
                    latest_count = 0

                batch_count += len(batch_labels)

            print("Epoch %d finished on %d batches in avg loss %f and cur loss %f." % (epoch, batch_count, total_loss/batch_step, latest_avg_loss))

            feed = {average_loss_holder: total_loss/batch_step}
            average_loss_str = sess.run(average_loss_tensor, feed_dict=feed)
            writer.add_summary(average_loss_str, epoch)

            feed1 = {cur_loss_holder: latest_avg_loss}
            cur_loss_str = sess.run(cur_loss_tensor, feed_dict=feed1)
            writer.add_summary(cur_loss_str, epoch)

            data_set.reset()
            if epoch % self.cfg.TRAIN.SAVE_STEPS == 0:
                filename = self.cfg.DIR.detector_net_saver_file_prefix + '{:d}'.format(epoch)
                filename = os.path.join(self.cfg.DIR.detector_net_saver_dir, filename)
                saver.save(sess, filename, global_step=epoch)

            if epoch % self.cfg.TRAIN.VALIDATE_EPOCHES == 0 and enable_validate:
                self.validate(sess=sess, writer=writer, epoch=epoch)
        filename = os.path.join(self.cfg.DIR.detector_net_saver_dir, (self.cfg.DIR.detector_net_saver_file_prefix
                                                                      + 'final'))
        saver.save(sess, filename)
        end_time = time.time()

        print("The total time used in training: {}".format(end_time-start_time))


    def build_model(self):

        self.X = tf.placeholder(tf.float32, shape=[None, 1, DIMEN_X, DIMEN_X, DIMEN_X])
        self.coord = tf.placeholder(tf.float32, shape=[None, 3, DIMEN_Y, DIMEN_Y, DIMEN_Y])
        self.labels = tf.placeholder(tf.float32, shape=[None, DIMEN_Y, DIMEN_Y, DIMEN_Y, 3, 5])
        self.condition = tf.placeholder(tf.float32, name="condition")


        self.net_config, self.detector_net_object, loss_object, self.pbb = get_model()

        self.feat, self.out = self.detector_net_object.getDetectorNet(self.X, self.coord)

        self.loss = loss_object.getLoss(self.out, self.labels, self.condition)

        tf.summary.histogram("Pred_bbox", self.out)
        tf.summary.histogram("Feature", self.feat)

        for key, value in self.detector_net_object.layers.items():
            tf.summary.histogram(key, value);

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.TRAIN.LEARNING_RATE).minimize(self.loss)

        self.summary_op = tf.summary.merge_all()

    def test(self, sess):
        sess.run(tf.global_variables_initializer())
        # load the previous trained detector_net model
        value_list = []
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope'))
        saver = tf.train.Saver(value_list, max_to_keep=100)
        saver.restore(sess, tf.train.latest_checkpoint(self.cfg.DIR.detector_net_saver_dir))
        # get input data
        split_combine = SplitComb(side_len=SIDE_LEN, max_stride=self.net_config['max_stride'],
                                  stride=self.net_config['stride'], margin=MARGIN,
                                  pad_value=self.net_config['pad_value'])
        input_data = TrainingDetectorData(data_dir=self.cfg.DIR.preprocess_result_path,
                                          split_path=self.cfg.DIR.detector_net_test_data_path,
                                          config=self.net_config, split_comber=split_combine,
                                          phase='test')
        total_nodules = 0
        predict_nodules = 0
        average_iou = 0
        for id in range(input_data.__len__()):
            imgs, bboxes, coord2, nzhw, filename = input_data.__getitem__(id)

            total_size_per_img = imgs.shape[0]

            index = 0
            final_out = None

            while index + self.cfg.TRAIN.BATCH_SIZE < total_size_per_img:
                feat_predict, out_predict = sess.run([self.feat, self.out], feed_dict={
                    self.X: imgs[index:index + self.cfg.TRAIN.BATCH_SIZE],
                    self.coord: coord2[index:index + self.cfg.TRAIN.BATCH_SIZE]})
                if final_out is None:
                    final_out = out_predict
                else:
                    final_out = np.concatenate((final_out, out_predict), axis=0)

                index = index + self.cfg.TRAIN.BATCH_SIZE

            if index < total_size_per_img:
                feat_predict, out_predict = sess.run([self.feat, self.out], feed_dict={
                    self.X: imgs[index:], self.coord: coord2[index:]})
                if final_out is None:
                    final_out = out_predict
                else:
                    final_out = np.concatenate((final_out, out_predict), axis=0)
            output = split_combine.combine(final_out, nzhw=nzhw)
            thresh = -3
            pbb, _ = self.pbb(output, thresh, ismask=True)
            pbb = pbb[pbb[:, 0] > self.cfg.TEST.DETECTOR_NODULE_CONFIDENCE]
            pbb = nms(pbb, self.cfg.TEST.DETECTOR_NODULE_OVERLAP)
            per_iou = 0
            for p in pbb:
                for l in bboxes:
                    score = iou(p[1:5], l)
                    if score > self.cfg.TEST.DETECTOR_NODULE_TH:
                        per_iou += score
                        predict_nodules += 1
                        break
            total_nodules += len(bboxes)
            average_iou += per_iou/len(bboxes)

        print("Total nodules:{}".format(total_nodules))
        print("Found nodules from Detector-Net:{}".format(predict_nodules))
        print("Average iou:{}".format(average_iou/input_data.__len__()))

    def validate(self, sess, writer, epoch):

        split_combine = SplitComb(side_len=SIDE_LEN, max_stride=self.net_config['max_stride'],
                                  stride=self.net_config['stride'], margin=MARGIN,
                                  pad_value=self.net_config['pad_value'])
        input_data = TrainingDetectorData(data_dir=self.cfg.DIR.preprocess_result_path,
                                          split_path=self.cfg.DIR.detector_net_test_data_path,
                                          config=self.net_config, split_comber=split_combine,
                                          phase='val')
        total_nodules = 0
        predict_nodules = 0
        average_iou = 0
        for id in range(input_data.__len__()):
            imgs, bboxes, coord2, nzhw, filename = input_data.__getitem__(id)

            total_size_per_img = imgs.shape[0]

            index = 0
            final_out = None

            while index + self.cfg.TRAIN.BATCH_SIZE < total_size_per_img:
                feat_predict, out_predict = sess.run([self.feat, self.out], feed_dict={
                    self.X: imgs[index:index + self.cfg.TRAIN.BATCH_SIZE],
                    self.coord: coord2[index:index + self.cfg.TRAIN.BATCH_SIZE]})
                if final_out is None:
                    final_out = out_predict
                else:
                    final_out = np.concatenate((final_out, out_predict), axis=0)

                index = index + self.cfg.TRAIN.BATCH_SIZE

            if index < total_size_per_img:
                feat_predict, out_predict = sess.run([self.feat, self.out], feed_dict={
                    self.X: imgs[index:], self.coord: coord2[index:]})
                if final_out is None:
                    final_out = out_predict
                else:
                    final_out = np.concatenate((final_out, out_predict), axis=0)
            output = split_combine.combine(final_out, nzhw=nzhw)
            thresh = -3
            pbb, _ = self.pbb(output, thresh, ismask=True)
            pbb = pbb[pbb[:, 0] > self.cfg.TEST.DETECTOR_NODULE_CONFIDENCE]
            pbb = nms(pbb, self.cfg.TEST.DETECTOR_NODULE_OVERLAP)
            per_iou = 0
            for p in pbb:
                for l in bboxes:
                    score = iou(p[1:5], l)
                    if score > self.cfg.TEST.DETECTOR_NODULE_TH:
                        per_iou += score
                        predict_nodules += 1
                        break
            total_nodules += len(bboxes)
            average_iou += per_iou/len(bboxes)

        feed = {self.validate_average_iou_holder: average_iou / input_data.__len__(),
                self.validate_nodule_predict_ratio_holder: predict_nodules/total_nodules}
        iou, ratio = sess.run(self.validate_average_iou_tensor, self.validate_nodule_predict_ratio_tensor,
                                    feed_dict=feed)
        writer.add_summary(iou, epoch)
        writer.add_summary(ratio, epoch)

    def predict(self, sess, splt_path):

        save_dir = os.path.join(self.cfg.DIR.bbox_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        sess.run(tf.global_variables_initializer())
        # load the previous trained detector_net model
        value_list = []
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope'))
        saver = tf.train.Saver(value_list, max_to_keep=100)
        saver.restore(sess, tf.train.latest_checkpoint(self.cfg.DIR.detector_net_saver_dir))

        # get input data
        split_combine = SplitComb(side_len=SIDE_LEN, max_stride=self.net_config['max_stride'],
                                  stride=self.net_config['stride'], margin=MARGIN,
                                  pad_value=self.net_config['pad_value'])
        input_data = TrainingDetectorData(data_dir=self.cfg.DIR.preprocess_result_path,
                                          split_path=splt_path,
                                          config=self.net_config, split_comber=split_combine,
                                          phase='test')
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
                    final_out = np.concatenate((final_out, out_predict), axis=0)

                index = index + self.cfg.TRAIN.BATCH_SIZE

            if index < total_size_per_img:
                feat_predict, out_predict = sess.run([self.feat, self.out], feed_dict={
                    self.X: imgs[index:], self.coord: coord2[index:]})
                if final_out is None:
                    final_out = out_predict
                else:
                    final_out = np.concatenate((final_out, out_predict), axis=0)
            print(final_out.shape)
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        instance.train(sess, continue_training=False)
        #instance.predict(sess, splt_path=cfg.DIR.detector_net_train_data_path)