import tensorflow as tf


class DetectorNetLoss():
    def __init__(self, num_hard=0):
        self.num_hard = num_hard

    def getLoss(self, output, labels, batch_size):
        output = tf.reshape(output, shape=(-1, 5))
        labels = tf.reshape(labels, shape= (-1, 5))


        pos_idcs = labels[:, 0] > 0.5
        pos_idcs_list=[]
        for i in range(5):
            pos_idcs_list.append(pos_idcs)
        pos_idcs = tf.stack(pos_idcs_list, axis=1)

        pos_output=tf.convert_to_tensor(tf.reshape(tf.boolean_mask(output,pos_idcs),(-1,5)))
        pos_labels=tf.convert_to_tensor(tf.reshape(tf.boolean_mask(labels,pos_idcs),(-1,5)))

        neg_idcs = labels[:, 0] < -0.5

        neg_output=tf.boolean_mask(output[:, 0], neg_idcs)
        neg_labels=tf.boolean_mask(labels[:, 0], neg_idcs)

        #if self.num_hard > 0 and train:
        #    neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        # Note: sigmoid_cross_entropy_with_logits already contains sigmoid, please do not sigmoid in advance,
        # which may cause loss not to decrease.
        #neg_prob = tf.nn.sigmoid(neg_output)
        neg_prob = neg_output

        pos_prob = tf.nn.sigmoid(pos_output[:, 0])
        pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
        lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

        regress_losses = [
            tf.losses.huber_loss(pz, lz),
            tf.losses.huber_loss(ph, lh),
            tf.losses.huber_loss(pw, lw),
            tf.losses.huber_loss(pd, ld)]
        classify_loss_with_pos_neg_without_hard_mining = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_prob, labels=pos_labels[:, 0])) + tf.reduce_mean(
            0.5 * tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_prob, labels=(neg_labels + 1)))

        classify_loss_without_pos_without_hard_mining = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_prob, labels=(neg_labels + 1)))

        classify_loss_without_neg = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_prob, labels=pos_labels[:, 0]))

        # create loss based on the hard_mining
        _, idcs = tf.nn.top_k(neg_output, self.num_hard * batch_size)
        neg_prob = tf.gather(neg_output, idcs)
        neg_labels = tf.gather(neg_labels, idcs)

        classify_loss_with_pos_neg_with_hard_mining = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_prob, labels=pos_labels[:, 0])) + tf.reduce_mean(
            0.5 * tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_prob, labels=(neg_labels + 1)))
        classify_loss_without_pos_with_hard_mining = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_prob, labels=(neg_labels + 1)))

        for regress_loss in regress_losses:
            classify_loss_with_pos_neg_without_hard_mining += regress_loss
            classify_loss_without_neg += regress_loss
            classify_loss_with_pos_neg_with_hard_mining += regress_loss

        return classify_loss_with_pos_neg_without_hard_mining, classify_loss_without_pos_without_hard_mining,\
               classify_loss_without_neg, classify_loss_with_pos_neg_with_hard_mining,\
               classify_loss_without_pos_with_hard_mining


if __name__ == "__main__":

    loss_object = DetectorNetLoss()
    labels = tf.placeholder(tf.float32, shape=[100, 32, 32, 32, 3, 5])
    output = tf.placeholder(tf.float32, shape=[100, 32, 32, 32, 3, 5])

    print(loss_object.getLoss(labels, output))


    with tf.Session() as sess:
        labels = tf.random_uniform(shape=[100,32,32,32,3,5])
        output = tf.random_uniform(shape=[100,32,32,32,3,5])

        print(sess.run(labels))
        sess.run(output)

        sess.run(loss_object.getLoss(labels, output))
