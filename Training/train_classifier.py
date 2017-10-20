import tensorflow as tf

from Training.Classifier.TrainingClassifierData import TrainingClassifierData
from Net.tensorflow_model.ClassiferNet import get_model
from Net.Classifer_Net_Loss import ClassiferNetLoss
from Training.configuration_training import cfg


class ClassiferTrainer(object):

    """
    Initializer
    """
    def __init__(self, cfg, detectorNet):

        self.cfg = cfg
        self.detectorNet = detectorNet
        self.build_model()



    def train(self, sess):

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(cfg.DIR.detector_net_saver_dir))

        dataset = TrainingClassifierData(cfg.DIR.preprocess_result_path,
                                         cfg.DIR.bbox_path,
                                         cfg.DIR.kaggle_full_labels,
                                         cfg.DIR.classifier_net_train_data_path,
                                         self.net_config,
                                         phase='train')

        for epoch in range(0, self.cfg.TRAIN.EPOCHS):

            batch_count = 1

            while dataset.hasNextBatch():

                batch_data, batch_coord, batch_isnode, batch_labels = dataset.getNextBatch(self.cfg.TRAIN.BATCH_SIZE)

                sess.run([self.loss_1_optimizer],
                         feed_dict={self.X: batch_data, self.coord: batch_coord,
                                    self.labels: batch_labels, self.isnod: batch_isnode})

                if batch_count % self.cfg.TRAIN.DISPLAY_STEPS:
                    print("Current batch is %d" % batch_count)

                batch_count += 1

            print("Epoch %d finished." % epoch)
            dataset.reset()

    def build_model(self):

        self.net_config, classifer_net_object = get_model(self.detectorNet)

        topK = self.net_config['topk']

        self.X = tf.placeholder(tf.float32, shape=[None, topK, 1, 128, 128, 128])
        self.coord = tf.placeholder(tf.float32, shape=[None, topK, 3, 32, 32, 32])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.isnod = tf.placeholder(tf.float32, shape=[None, topK])

        nodulePred, casePred, casePred_each = classifer_net_object.getClassiferNet(self.X, self.coord)

        loss_object = ClassiferNetLoss(self.net_config)
        self.loss = loss_object.getLoss(casePred, casePred_each, self.labels, self.isnod, self.cfg.TRAIN.BATCH_SIZE, topK)

        self.loss_1_optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.TRAIN.LEARNING_RATE).minimize(self.loss)


    def test(self):
        pass


    def validate(self):
        pass


if __name__ == "__main__":


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        detectorNet = None
        instance = ClassiferTrainer(cfg, detectorNet)
        sess.run(init)
        #instance.train(sess)

