import tensorflow as tf

def hard_mining(neg_output, neg_labels, num_hard):
    _, indcs = tf.nn.top_k(neg_output, tf.minimum(num_hard, tf.shape(neg_output)[0]))
    neg_output = tf.gather(neg_output, indcs)
    neg_labels = tf.gather(neg_labels, indcs)
    return neg_output, neg_labels


def safe_sce_loss(logits, labels):
    return tf.cond(tf.less(tf.constant(0), tf.shape(logits)[0]),
                   lambda: tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
                   lambda: tf.constant(0.))


class PosClassifyLoss():
    def __init__(self):
        pass

    def __call__(self, arg_pos_output, arg_pos_labels, arg_neg_output, arg_neg_labels):
        classify_loss = tf.reduce_mean(0.5 * safe_sce_loss(logits=arg_pos_output[:, 0], labels=arg_pos_labels[:, 0])) \
                        + tf.reduce_mean(0.5 * safe_sce_loss(logits=arg_neg_output, labels=(arg_neg_labels + 1)))

        return classify_loss

class PosL1Loss():
    def __init__(self):
        pass

    def __call__(self, arg_pos_output, arg_pos_labels):
        pz, ph, pw, pd = arg_pos_output[:, 1], arg_pos_output[:, 2], arg_pos_output[:, 3], arg_pos_output[:, 4]
        lz, lh, lw, ld = arg_pos_labels[:, 1], arg_pos_labels[:, 2], arg_pos_labels[:, 3], arg_pos_labels[:, 4]

        regress_losses = [
            tf.losses.huber_loss(lz, pz),
            tf.losses.huber_loss(lh, ph),
            tf.losses.huber_loss(lw, pw),
            tf.losses.huber_loss(ld, pd)
        ]

        return regress_losses

class NegClassifyLoss():
    def __init__(self):
        pass

    def __call__(self, arg_neg_output, arg_neg_labels):
        classify_loss = tf.reduce_mean(0.5 * safe_sce_loss(logits=arg_neg_output, labels=(arg_neg_labels+1)))
        return classify_loss

class NegL1Loss():
    def __init__(self):
        pass

    def __call__(self):
        regress_losses = [0., 0., 0., 0.]
        return regress_losses


class DetectorNetLoss():

    def __init__(self, num_hard=0):
        self.num_hard = num_hard
        self.pos_classify_loss = PosClassifyLoss()
        self.pos_l1_loss = PosL1Loss()
        self.neg_classify_loss = NegClassifyLoss()
        self.neg_l1_loss = NegL1Loss()


    def getLoss(self, output, labels):

        batch_size = labels.shape[0]
        output = tf.reshape(output, shape=(-1, 5))
        labels = tf.reshape(labels, shape=(-1, 5))

        pos_output = tf.boolean_mask(output, labels[:, 0] > 0.5)
        pos_labels = tf.boolean_mask(labels, labels[:, 0] > 0.5)

        neg_output = tf.boolean_mask(output[:, 0], labels[:, 0] < -0.5)
        neg_labels = tf.boolean_mask(labels[:, 0], labels[:, 0] < -0.5)

        if self.num_hard > 0:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard)
        #neg_prob = tf.nn.sigmoid(neg_output)

        classify_loss = tf.cond(tf.less(tf.constant(0), tf.shape(pos_output)[0]),
                      lambda: self.pos_classify_loss(pos_output, pos_labels, neg_output, neg_labels),
                      lambda: self.neg_classify_loss(neg_output, neg_labels))

        regress_losses = tf.cond(tf.less(tf.constant(0), tf.shape(pos_output)[0]),
                      lambda: self.pos_l1_loss(pos_output, pos_labels),
                      lambda: self.neg_l1_loss())

        return regress_losses + [classify_loss]



if __name__ == "__main__":

    loss_object = DetectorNetLoss()
    labels = tf.placeholder(tf.float32, shape=[100, 32, 32, 32, 3, 5])
    output = tf.placeholder(tf.float32, shape=[100, 32, 32, 32, 3, 5])

    print(loss_object.getLoss(labels, output))


    with tf.Session() as sess:
        labels = tf.random_uniform(shape=[100,32,32,32,3,5])
        output = tf.random_uniform(shape=[100,32,32,32,3,5])

        sess.run(labels)
        sess.run(output)

        l1, l2, l3, l4, cls = sess.run(loss_object.getLoss(labels, output))

        print(l1)
        print(cls)

