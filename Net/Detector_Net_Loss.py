import tensorflow as tf


def hard_mining(neg_output, neg_labels, num_hard):
    size = neg_output.get_shape().as_list()
    _, idcs = tf.nn.top_k(neg_output, tf.minimum(num_hard, size[0]))
    neg_output = tf.gather(neg_output, idcs)
    neg_labels = tf.gather(neg_labels, idcs)
    return neg_output, neg_labels


class DetectorNetLoss():
    def __init__(self, num_hard=0):
        self.num_hard = num_hard

    def getLoss(self, output, labels, train=True):
        batch_size = labels.get_shape().as_list()
        output = tf.reshape(output, shape=(-1, 5))
        labels = tf.reshape(labels, shape= (-1, 5))


        pos_idcs = labels[:, 0] > 0.5
        #print(pos_idcs.shape)
        #pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_idcs_list=[]
        for i in range(5):
            pos_idcs_list.append(pos_idcs)
        pos_idcs = tf.stack(pos_idcs_list, axis=1)

        #print(pos_idcs.shape)
        #pos_output = tf.reshape(output[pos_idcs], shape=(-1, 5))
        #print(tf.boolean_mask(output,pos_idcs).eval())

        pos_output=tf.convert_to_tensor(tf.reshape(tf.boolean_mask(output,pos_idcs),(-1,5)))
        #pos_labels = tf.reshape(labels[pos_idcs], shape=(-1, 5))
        pos_labels=tf.convert_to_tensor(tf.reshape(tf.boolean_mask(labels,pos_idcs),(-1,5)))

        #print(pos_output.eval().shape[0])

        neg_idcs = labels[:, 0] < -0.5
        #neg_output = output[:, 0][neg_idcs]
        #print(output[:,0].shape)
        neg_output=tf.boolean_mask(output[:,0], neg_idcs)
        #neg_labels = labels[:, 0][neg_idcs]
        neg_labels=tf.boolean_mask(labels[:, 0], neg_idcs)

        #if self.num_hard > 0 and train:
        #    neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = tf.nn.sigmoid(neg_output)


        #if pos_output.shape[0] > 0:
        pos_prob = tf.nn.sigmoid(pos_output[:, 0])
        pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
        lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

        regress_losses = [
            tf.losses.huber_loss(pz, lz),
            tf.losses.huber_loss(ph, lh),
            tf.losses.huber_loss(pw, lw),
            tf.losses.huber_loss(pd, ld)]
        #regress_losses_data_1 = [l[0] for l in regress_losses]
        classify_loss_1 = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_prob, labels=pos_labels[:, 0])) + tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_prob, labels=(neg_labels + 1)))
        pos_correct = tf.reduce_sum(tf.cast(pos_prob >= 0.5, dtype=tf.int32))
        pos_total = pos_prob.shape[0]

        # else
        #regress_losses_2 = [0, 0, 0, 0]
        classify_loss_2 = tf.reduce_mean(0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_prob, labels=(neg_labels + 1)))
        #pos_correct = 0
        #pos_total = 0
        #regress_losses_data = [0, 0, 0, 0]


        #classify_loss_data = classify_loss_[0]

        loss1 = classify_loss_1
        loss2 = classify_loss_2
        for regress_loss in regress_losses:
            loss1 += regress_loss


        neg_correct = tf.reduce_sum(tf.cast(neg_prob < 0.5, dtype=tf.int32))
        neg_total = neg_prob.shape[0]

        return loss1, loss2
        #return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]


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
