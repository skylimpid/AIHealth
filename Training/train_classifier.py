import os
import time
import shutil
import tensorflow as tf

from Training.Classifier.training_classifier_data import TrainingClassifierData
from Net.tensorflow_model.classifier_net import get_model
from Net.classifier_net_loss import ClassifierNetLoss
from Training.configuration_training import cfg
from Net.tensorflow_model.detector_net import DetectorNet
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

        #sess.run(tf.global_variables_initializer())

        average_loss_holder = tf.placeholder(tf.float32)
        average_loss_tensor = tf.summary.scalar("cl_loss", average_loss_holder)

        average_accuracy_holder = tf.placeholder(tf.float32)
        average_accuracy_tensor = tf.summary.scalar("cl_accuracy", average_accuracy_holder)

        writer = tf.summary.FileWriter(CLASSIFIER_NET_TENSORBOARD_LOG_DIR)
        writer.add_graph(sess.graph)

        saver = tf.train.Saver(max_to_keep=100)

        if not os.path.exists(cfg.DIR.classifier_net_saver_dir):
            os.makedirs(cfg.DIR.classifier_net_saver_dir)

        var_classifier = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/cl_scope')
        sess.run(tf.variables_initializer(var_classifier))

        dataset = TrainingClassifierData(cfg.DIR.preprocess_result_path,
                                         cfg.DIR.bbox_path,
                                         cfg.DIR.kaggle_full_labels,
                                         cfg.DIR.classifier_net_train_data_path,
                                         self.net_config,
                                         phase='train')

        start_time = time.time()
        tf.get_default_graph().finalize()

        for epoch in range(1, self.cfg.TRAIN_CL.EPOCHS+1):

            batch_count = 0
            total_loss = 0
            total_accuracy = 0
            while dataset.hasNextBatch():

                batch_data, batch_coord, batch_isnode, batch_labels = dataset.getNextBatch(self.cfg.TRAIN_CL.BATCH_SIZE)
                #print("batch_isnode shape: ", batch_isnode.shape)
                print("batch_labels : ", batch_labels)
                # batch_labels = batch_labels.reshape((-1, 1))
                print("batch_data: ")
                #print(batch_data)

                _, loss, accuracy_op, loss2, miss_loss, case_pred, case_pred_each, miss_mask, center_feat, nodule_feat = sess.run([self.loss_optimizer, self.loss, self.accuracy,
                                                 self.loss2,
                                                 self.miss_loss,
                                                 self.casePred,
                                                 self.casePred_each,
                                                 self.miss_mask,
                                                 self.center_feat,
                                                 self.nodule_feat
                                                                    ],
                                                feed_dict={self.X: batch_data,
                                                           self.coord: batch_coord,
                                                           self.labels: batch_labels,
                                                           self.isnod: batch_isnode})

                batch_count += 1
                total_loss += loss
                total_accuracy += accuracy_op

                if batch_count % self.cfg.TRAIN_CL.DISPLAY_STEPS:
                    # print("batch_labels:", batch_labels)
                    # print(loss2, miss_loss)
                    print("case_pred: ", case_pred)
                    print("case_pred_each: ", case_pred_each)
                    # print("miss mask: ", miss_mask)
                    # print("batch_is_nod: ", batch_isnode)
                    print("center_feat: ", center_feat)
                    #print("nodule_feat: ", nodule_feat)
                    print("Current batch: %d, loss: %f, accuracy: %f" % (batch_count, loss, accuracy_op))

            print("Epoch %d finished in loss: %f and accuracy: %f" % (epoch, total_loss/batch_count, total_accuracy/batch_count))
            feed = {average_loss_holder: total_loss/batch_count, average_accuracy_holder: total_accuracy/batch_count}
            average_loss_str, average_accuracy_str = sess.run([average_loss_tensor, average_accuracy_tensor],
                                                              feed_dict=feed)

            writer.add_summary(average_loss_str, epoch)
            writer.add_summary(average_accuracy_str, epoch)
            dataset.reset()
            if epoch % self.cfg.TRAIN_CL.SAVE_STEPS == 0:
                filename = self.cfg.DIR.classifier_net_saver_file_prefix + '{:d}'.format(epoch)
                filename = os.path.join(self.cfg.DIR.classifier_net_saver_dir, filename)
                saver.save(sess, filename, global_step=epoch)

        filename = os.path.join(self.cfg.DIR.classifier_net_saver_dir, (self.cfg.DIR.classifier_net_saver_file_prefix
                                                                        + 'final'))
        saver.save(sess, filename)
        end_time = time.time()

        print("The total time used in training: {}".format(end_time - start_time))

    def build_model(self):

        self.net_config, classifier_net_object = get_model(self.detectorNet)

        topK = self.net_config['topk']

        self.X = tf.placeholder(tf.float32, shape=[None, topK, 1, DIMEN_X, DIMEN_X, DIMEN_X])
        self.coord = tf.placeholder(tf.float32, shape=[None, topK, 3, DIMEN_Y, DIMEN_Y, DIMEN_Y])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.isnod = tf.placeholder(tf.float32, shape=[None, topK])

        self.nodulePred, self.casePred, self.casePred_each, self.center_feat, self.nodule_feat = classifier_net_object.get_classifier_net(self.X, self.coord)


        loss_object = ClassifierNetLoss(self.net_config)

        self.loss, self.loss2, self.miss_loss, self.miss_mask = loss_object.getLoss(self.casePred, self.casePred_each, self.labels, self.isnod, self.cfg.TRAIN_CL.BATCH_SIZE, topK)

        global_step = tf.Variable(0, trainable=False, name="cl_global_step")

        lr = tf.train.exponential_decay(self.cfg.TRAIN_CL.LEARNING_RATE,
                                        global_step,
                                        self.cfg.TRAIN_CL.LEARNING_RATE_STEP_SIZE,
                                        self.cfg.TRAIN_CL.LEARNING_RATE_DECAY_RATE,
                                        staircase=True)

        #self.loss_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.cfg.TRAIN_CL.MOMENTUM).minimize(
        #    self.loss, global_step=global_step)

        self.loss_optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(self.loss, global_step=global_step)

        correct_predict = tf.equal(self.labels[:,0], tf.cast(self.casePred >= 0.5, tf.float32))
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
                print('Variables to be restored: %s' % v.name)
                variables_to_restore.append(v)
        return variables_to_restore


    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        detector_net = DetectorNet()
        instance = ClassifierTrainer(cfg, detector_net)
        sess.run(tf.global_variables_initializer())
        variables = tf.global_variables()
        var_keep_dic = get_variables_in_checkpoint_file(tf.train.latest_checkpoint(cfg.DIR.detector_net_saver_dir))
        restorer = tf.train.Saver(get_variables_to_restore(variables, var_keep_dic))
        # var_detector = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope')
        # restorer = tf.train.Saver(var_detector)

        # print("before restore:")
        # print(tf.get_default_graph().get_tensor_by_name(
        #    name='global/detector_scope/global/detector_scope/resBlock6/resBlock6-3_conv2_bn/gamma/Adadelta_1:0').eval())

        restorer.restore(sess, tf.train.latest_checkpoint(cfg.DIR.detector_net_saver_dir))

        # print("after:")
        # print(tf.get_default_graph().get_tensor_by_name(name='global/detector_scope/global/detector_scope/resBlock6/resBlock6-3_conv2_bn/gamma/Adadelta_1:0').eval())

        instance.train(sess)

