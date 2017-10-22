import tensorflow as tf

from Training.Classifier.TrainingClassifierData import TrainingClassifierData
from Net.tensorflow_model.ClassiferNet import get_model
from Net.Classifer_Net_Loss import ClassiferNetLoss
from Training.configuration_training import cfg
from Net.tensorflow_model.DetectorNet import DecetorNet
import os
import time


class ClassifierTrainer(object):

    """
    Initializer
    """
    def __init__(self, cfg, detectorNet):

        self.cfg = cfg
        self.detectorNet = detectorNet
        self.build_model()

    def train(self, sess):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/home/xuan/tensorboard/1')
        writer.add_graph(sess.graph)

        value_list = []
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope'))
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/classifier_scope'))
        saver = tf.train.Saver(value_list, max_to_keep=100)

        var_classifier = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/classifier_scope')
        sess.run(tf.variables_initializer(var_classifier))


        sess.run(tf.global_variables_initializer())
        dataset = TrainingClassifierData(cfg.DIR.preprocess_result_path,
                                         cfg.DIR.bbox_path,
                                         cfg.DIR.kaggle_full_labels,
                                         cfg.DIR.classifier_net_train_data_path,
                                         self.net_config,
                                         phase='train')
        start_time = time.time()
        index = 1
        for epoch in range(0, self.cfg.TRAIN.EPOCHS):

            batch_count = 1

            while dataset.hasNextBatch():

                batch_data, batch_coord, batch_isnode, batch_labels = dataset.getNextBatch(self.cfg.TRAIN.BATCH_SIZE)

                merged_output,_, loss, accuracy_op = sess.run([merged,self.loss_1_optimizer, self.loss, self.accuracy],
                         feed_dict={self.X: batch_data, self.coord: batch_coord,
                                    self.labels: batch_labels, self.isnod: batch_isnode})
                writer.add_summary(merged_output, index)
                if batch_count % self.cfg.TRAIN.DISPLAY_STEPS:
                    print("Current batch is %d" % batch_count)

                batch_count += 1
                index += 1
                print("The loss is:{}".format(loss))
            print("Epoch %d finished." % epoch)
            dataset.reset()
            if epoch != 0 and epoch % self.cfg.TRAIN.SAVE_STEPS == 0:
                filename = self.cfg.DIR.classifier_net_saver_file_prefix + '{:d}'.format(epoch+1)
                filename = os.path.join(self.cfg.DIR.classifier_net_saver_dir, filename)
                saver.save(sess, filename, global_step=(epoch+1))

        filename = os.path.join(self.cfg.DIR.classifier_net_saver_dir, (self.cfg.DIR.classifier_net_saver_file_prefix
                                                                        + 'final'))
        saver.save(sess, filename)
        end_time = time.time()

        print("The total time used in training: {}".format(end_time-start_time))

    def build_model(self):

        self.net_config, classifer_net_object = get_model(self.detectorNet)

        topK = self.net_config['topk']

        self.X = tf.placeholder(tf.float32, shape=[None, topK, 1, 128, 128, 128])
        self.coord = tf.placeholder(tf.float32, shape=[None, topK, 3, 32, 32, 32])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.isnod = tf.placeholder(tf.float32, shape=[None, topK])

        nodulePred, casePred, casePred_each = classifer_net_object.getClassiferNet(self.X, self.coord)

        loss_object = ClassiferNetLoss(self.net_config)

        with tf.name_scope('loss'):
            self.loss = loss_object.getLoss(casePred, casePred_each, self.labels, self.isnod, self.cfg.TRAIN.BATCH_SIZE, topK)
        tf.summary.scalar('loss', self.loss)
        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(self.cfg.TRAIN.LEARNING_RATE, global_step,
                                        self.cfg.TRAIN.LEARNING_RATE_STEP_SIZE, 0.1, staircase=True)

        self.loss_1_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.cfg.TRAIN.MOMENTUM).minimize(
            self.loss, global_step=global_step)

        correct_predict = tf.equal(self.labels[:,0], tf.cast(casePred >= 0.5, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)


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

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        detectorNet = DecetorNet()
        instance = ClassifierTrainer(cfg, detectorNet)
        variables = tf.global_variables()
        sess.run(init)
        var_keep_dic = get_variables_in_checkpoint_file(tf.train.latest_checkpoint(cfg.DIR.detector_net_saver_dir))
        restorer = tf.train.Saver(get_variables_to_restore(variables, var_keep_dic))
        restorer.restore(sess, tf.train.latest_checkpoint(cfg.DIR.detector_net_saver_dir))
        instance.train(sess)

