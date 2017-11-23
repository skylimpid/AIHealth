import os
import time
import shutil
import tensorflow as tf

from Training.Classifier.training_classifier_data import TrainingClassifierData
from Net.tensorflow_model.classifier_net import get_model
from Net.classifier_net_loss import ClassifierNetLoss
from Training.configuration_training import cfg
from Net.tensorflow_model.detector_net import DecetorNet
from Training.constants import CLASSIFIER_NET_TENSORBOARD_LOG_DIR, DIMEN_X, DIMEN_Y


class ClassifierTrainer(object):

    """
    Initializer
    """
    def __init__(self, cfg, detectorNet):

        self.cfg = cfg
        self.detectorNet = detectorNet
        self.build_model()

    def train(self, sess, clear=True):

        if clear:
            if os.path.exists(CLASSIFIER_NET_TENSORBOARD_LOG_DIR):
                shutil.rmtree(CLASSIFIER_NET_TENSORBOARD_LOG_DIR)

        sess.run(tf.global_variables_initializer())

        average_loss_holder = tf.placeholder(tf.float32)
        average_loss_tensor = tf.summary.scalar("training_loss", average_loss_holder)

        average_accuracy_holder = tf.placeholder(tf.float32)
        average_accuracy_tensor = tf.summary.scalar("training_accuracy", average_accuracy_holder)

        writer = tf.summary.FileWriter(CLASSIFIER_NET_TENSORBOARD_LOG_DIR)
        writer.add_graph(sess.graph)

        saver = tf.train.Saver(max_to_keep=100)

        if not os.path.exists(cfg.DIR.classifier_net_saver_dir):
            os.makedirs(cfg.DIR.classifier_net_saver_dir)

        var_classifier = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/classifier_scope')
        sess.run(tf.variables_initializer(var_classifier))

        dataset = TrainingClassifierData(cfg.DIR.preprocess_result_path,
                                         cfg.DIR.bbox_path,
                                         cfg.DIR.kaggle_full_labels,
                                         cfg.DIR.classifier_net_train_data_path,
                                         self.net_config,
                                         phase='train')

        start_time = time.time()
        tf.get_default_graph().finalize()

        for epoch in range(1, self.cfg.TRAIN.EPOCHS+1):

            batch_count = 0
            total_loss = 0
            total_accuracy = 0
            while dataset.hasNextBatch():

                batch_data, batch_coord, batch_isnode, batch_labels = dataset.getNextBatch(self.cfg.TRAIN.BATCH_SIZE)

                _, loss, accuracy_op = sess.run([self.loss_1_optimizer, self.loss, self.accuracy],
                                                feed_dict={self.X: batch_data,
                                                           self.coord: batch_coord,
                                                           self.labels: batch_labels,
                                                           self.isnod: batch_isnode})

                batch_count += 1
                total_loss += loss
                total_accuracy += accuracy_op

                if batch_count % self.cfg.TRAIN.DISPLAY_STEPS:
                    print("Current batch: %d, loss: %f, accuracy: %f" % (batch_count, loss, accuracy_op))

            print("Epoch %d finished in loss: %f and accuracy: %f" % (epoch, total_loss/batch_count, total_accuracy/batch_count))
            feed = {average_loss_holder: total_loss/batch_count, average_accuracy_holder: total_accuracy/batch_count}
            average_loss_str, average_accuracy_str = sess.run([average_loss_tensor, average_accuracy_tensor],
                                                              feed_dict=feed)

            writer.add_summary(average_loss_str, epoch)
            writer.add_summary(average_accuracy_str, epoch)
            dataset.reset()
            if epoch % self.cfg.TRAIN.SAVE_STEPS == 0:
                filename = self.cfg.DIR.classifier_net_saver_file_prefix + '{:d}'.format(epoch)
                filename = os.path.join(self.cfg.DIR.classifier_net_saver_dir, filename)
                saver.save(sess, filename, global_step=epoch)

        filename = os.path.join(self.cfg.DIR.classifier_net_saver_dir, (self.cfg.DIR.classifier_net_saver_file_prefix
                                                                        + 'final'))
        saver.save(sess, filename)
        end_time = time.time()

        print("The total time used in training: {}".format(end_time-start_time))

    def build_model(self):

        self.net_config, classifier_net_object = get_model(self.detectorNet)

        topK = self.net_config['topk']

        self.X = tf.placeholder(tf.float32, shape=[None, topK, 1, DIMEN_X, DIMEN_X, DIMEN_X])
        self.coord = tf.placeholder(tf.float32, shape=[None, topK, 3, DIMEN_Y, DIMEN_Y, DIMEN_Y])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.isnod = tf.placeholder(tf.float32, shape=[None, topK])

        nodulePred, casePred, casePred_each = classifier_net_object.get_classifier_net(self.X, self.coord)

        loss_object = ClassifierNetLoss(self.net_config)

        self.loss = loss_object.getLoss(casePred, casePred_each, self.labels, self.isnod, self.cfg.TRAIN.BATCH_SIZE, topK)

        global_step = tf.Variable(0, trainable=False, name="classifier_global_step")

        lr = tf.train.exponential_decay(self.cfg.TRAIN.LEARNING_RATE, global_step,
                                        self.cfg.TRAIN.LEARNING_RATE_STEP_SIZE, 0.1, staircase=True)

        self.loss_1_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.loss, global_step=global_step)

        correct_predict = tf.equal(self.labels[:,0], tf.cast(casePred >= 0.5, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    def test(self):
        pass


    def validate(self):
        pass


if __name__ == "__main__":

    def get_variables_in_checkpoint_file(file_name):
        try:
            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))

    def get_variables_to_restore(variables, var_keep_dic):
        variables_to_restore = []
        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)
        return variables_to_restore

    with tf.Session() as sess:
        detectorNet = DecetorNet()
        instance = ClassifierTrainer(cfg, detectorNet)
        variables = tf.global_variables()
        var_keep_dic = get_variables_in_checkpoint_file(tf.train.latest_checkpoint(cfg.DIR.detector_net_saver_dir))
        restorer = tf.train.Saver(get_variables_to_restore(variables, var_keep_dic))
        # var_detector = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope')
        # restorer = tf.train.Saver(var_detector)
        # restorer.restore(sess, tf.train.latest_checkpoint(cfg.DIR.detector_net_saver_dir))
        instance.train(sess)

