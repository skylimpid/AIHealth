import tensorflow as tf

from Training.Detector.TrainingDetectorData import TrainingDetectorData

class DetectorTrainer(object):


    """
    Initializer

    """
    def __init__(self, config):

        # TO-DO: Initialize more parameters
        self.config = config

        # TO-DO: fill in the arguments in the initializer.
        self.dataset = TrainingDetectorData()

        self.optimizer, self.loss, self.gradient = self.build_model()



    def train(self, sess):

        for epoch in range(0, self.config['num_of_epoch']):

            while self.dataset.has_next():

                batch_data = self.dataset.getNextBatch(self.config['batch_size'])

                ## Probably needs clip gradients, but CNN might be fine without
                ## Probably needs dropout
                loss_value = sess.run([self.loss, self.optimizer],
                                      feed_dict={self.input_batch:batch_data})


    def build_model(self):
        pass

    def test(self):
        pass


    def validate(self):
        pass
