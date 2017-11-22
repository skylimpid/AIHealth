import tensorflow as tf
import numpy as np

config = {}
config['bbox_factor'] = 10
config['pos_factor'] = 5
config['neg_factor'] = 1


"""
Inspired by YOLO
 -- pre-define the penalty factors for pos_probality, neg_probality, and bbox
 -- use IOU to calcuate bboxes difference.
"""
class DetectorNetLoss_YOLO():

    def __init__(self):
        pass

    def iou_tf(self, box_labels, box_pred):
        z_int = tf.minimum(box_labels[1] + 0.5 * box_labels[4], box_pred[1] + 0.5 * box_pred[4])\
                - tf.maximum(box_labels[1] - 0.5 * box_labels[4], box_pred[1] - 0.5 * box_pred[4])
        y_int = tf.minimum(box_labels[2] + 0.5 * box_labels[4], box_pred[2] + 0.5 * box_pred[4])\
                - tf.maximum(box_labels[2] - 0.5 * box_labels[4], box_pred[2] - 0.5 * box_pred[4])
        x_int = tf.minimum(box_labels[3] + 0.5 * box_labels[4], box_pred[3] + 0.5 * box_pred[4])\
                - tf.maximum(box_labels[3] - 0.5 * box_labels[4],box_pred[3] - 0.5 * box_pred[4])
        intersection = tf.multiply(x_int, y_int)
        intersection = tf.multiply(intersection, z_int)

        neg_x_int = tf.less(x_int, 0)
        neg_y_int = tf.less(y_int, 0)
        neg_z_int = tf.less(z_int, 0)

        neg_wht = tf.less(box_pred[4], 0)

        neg_xy = tf.logical_or(neg_x_int, neg_y_int)
        neg_xyz = tf.logical_or(neg_xy, neg_z_int)

        neg = tf.logical_or(neg_xyz, neg_wht)

        pos = tf.cast(tf.logical_not(neg), tf.float32)
        intersection = tf.multiply(intersection, pos)

        vol1 = tf.multiply(box_labels[4], box_labels[4])
        vol1 = tf.multiply(vol1, box_labels[4])

        vol2 = tf.multiply(box_pred[4], box_pred[4])
        vol2 = tf.multiply(vol2, box_pred[4])

        zero_vol1 = tf.equal(vol1, tf.zeros_like(vol1))
        vol1 = vol1 + tf.cast(zero_vol1, tf.float32)

        # account for possible negative volume in box_pred
        vol2 = tf.abs(vol2)

        # account for tiny box in vol2 ( < eps )
        eps = 1e-7
        zero_vol2 = tf.less_equal(vol2, eps * tf.ones_like(vol2))
        vol2 = vol2 + tf.cast(zero_vol2, tf.float32)

        union = vol1 + vol2 - intersection
        return tf.divide(intersection, union)


    def loss_pos_neg(self, output, labels, num_pos):
        pos_idcs = labels[:, 0] > 0.5
        pos_output = tf.boolean_mask(output, pos_idcs)
        pos_labels = tf.boolean_mask(labels, pos_idcs)

        neg_idcs = labels[:, 0] < -0.5
        neg_output=tf.boolean_mask(output[:,0], neg_idcs)
        neg_labels=tf.boolean_mask(labels[:, 0], neg_idcs)


        regress_losses = num_pos - tf.reduce_sum(self.iou_tf(
            box_labels = pos_labels[:, 1:], box_pred = pos_output[:, 1:]))

        pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pos_output[:, 0], labels=pos_labels[:, 0]))

        neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=neg_output, labels=(neg_labels + 1)))

        return config['bbox_factor'] * regress_losses + config['pos_factor'] * pos_loss\
               + config['neg_factor'] * neg_loss

    def loss_neg(self, output, labels):
        neg_labels = labels[:, 0]
        return config['neg_factor'] * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=output[:,0], labels=(neg_labels + 1)))



    def getLoss(self, output, labels, num_pos):
        output = tf.reshape(output, shape=(-1, 5))
        labels = tf.reshape(labels, shape= (-1, 5))
        return tf.cond(num_pos > 0, lambda:self.loss_pos_neg(output, labels, num_pos),
                       lambda:self.loss_neg(output, labels))


if __name__ == "__main__":

    loss_object = DetectorNetLoss_YOLO()
    labels = tf.placeholder(tf.float32, shape=[100, 32, 32, 32, 3, 5])
    output = tf.placeholder(tf.float32, shape=[100, 32, 32, 32, 3, 5])
    condition = tf.placeholder(tf.float32, name="condition")
    loss_1 = loss_object.getLoss(output=output, labels=labels, num_pos=condition)

    with tf.Session() as sess:
        batch_labels = tf.random_uniform(shape=[100,32,32,32,3,5], minval=-1)
        batch_output = tf.random_uniform(shape=[100,32,32,32,3,5], minval=-1)
        loss = sess.run(loss_1, feed_dict={output:batch_output.eval(),labels:batch_labels.eval(), condition:2})
        print(loss)
