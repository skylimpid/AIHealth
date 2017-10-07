import tensorflow as tf


def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = tf.nn.top_k(neg_output, min(num_hard, len(neg_output)))
    neg_output = tf.gather(neg_output, idcs)
    neg_labels = tf.gather(neg_labels, idcs)
    return neg_output, neg_labels


class Loss():
    def __init__(self, num_hard=0):
        self.num_hard = num_hard

    def getLoss(self, output, labels, train=True):
        batch_size = labels.get_shape().as_list()
        output = tf.reshape(output, shape=(-1, 5))
        labels = tf.reshape(labels, shape= (-1, 5))

        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = tf.reshape(output[pos_idcs], shape=(-1, 5))
        pos_labels = tf.reshape(labels[pos_idcs], shape=(-1, 5))

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = tf.nn.sigmoid(neg_output)

        if len(pos_output) > 0:
            pos_prob = tf.nn.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                tf.losses.huber_loss(pz, lz),
                tf.losses.huber_loss(ph, lh),
                tf.losses.huber_loss(pw, lw),
                tf.losses.huber_loss(pd, ld)]
            regress_losses_data = [l.data[0] for l in regress_losses]
            classify_loss = 0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                pos_prob, pos_labels[:, 0]) + 0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0, 0, 0, 0]
            classify_loss = 0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0, 0, 0, 0]
        classify_loss_data = classify_loss.data[0]

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]