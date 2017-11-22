from Net.keras_model.detector_net_keras import DetectNet
import keras.backend as K
from keras.layers import Lambda, MaxPooling3D, Dropout, Dense, Activation
import tensorflow as tf

class ClassiferNet():
    def __init(self, topK):
        self.NoduleNet = DetectNet()
        self.topK = topK

    def createOutputNode(self, inputs):
        xlist = inputs[0]

        xsize = K.int_shape(xlist)
        output_shape=(-1, xsize[2], xsize[3], xsize[4], xsize[5])
        xlist = Lambda(self.postprocessTensor,
                        output_shape=output_shape)(xlist, output_shape)

        coordlist = inputs[1]
        corrdsize = K.int_shape(coordlist)
        output_shape=(-1, corrdsize[2], corrdsize[3], corrdsize[4], corrdsize[5])
        coordlist = Lambda(self.postprocessTensor,
                        output_shape=output_shape)(coordlist, output_shape)

        noduleFeat, nodulePred = self.NoduleNet(xlist, coordlist)

        output_shape = (corrdsize[0],corrdsize[1],-1)
        nodulePred = Lambda(self.postprocessTensor,
                        output_shape=output_shape)(nodulePred, output_shape)

        featshape = K.int_shape(noduleFeat)
        centerFeat = MaxPooling3D()(pool_size=2, strides=2, padding="valid")(noduleFeat[:,:,featshape[2]/2-1:featshape[2]/2+1,
                                          featshape[3]/2-1:featshape[3]/2+1,
                                          featshape[4]/2-1:featshape[4]/2+1])
        centerFeat = centerFeat[:, :, 0, 0, 0]
        out = Dropout(0.5)(centerFeat)
        fc1 = Dense(64, name="fc1")(out)
        fc1_active = Activation(K.relu)(fc1)
        fc2 = Dense(1, name="fc2")(fc1_active)
        fc2_active = Activation(K.sigmoid)(fc2)
        output_shape = (xsize[0],xsize[1])
        out = Lambda(self.postprocessTensor,
                    output_shape=output_shape)(fc2_active, output_shape)
        base_prob = tf.sigmoid(tf.constant(-30.0, dtype=tf.float32))
        casePred = 1-K.prod(1-out, axis=1)*(1-base_prob.expand(out.size()[0]))
        return nodulePred,casePred,out



    def postprocessTensor(self, xlist, outputshape):
        out = K.reshape(xlist, outputshape)
        return out