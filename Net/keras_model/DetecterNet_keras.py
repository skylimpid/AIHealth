import numpy as np
import Net.keras_model
from keras.models import Model, Input
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Activation, Dropout, Lambda, merge
from keras import backend as K
from keras_contrib.layers import Deconvolution3D
import keras

config = {}
config['anchors'] = [10,30,60]

'''
preBlock): Sequential (
    (0): Conv3d(1, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (1): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU (inplace)
    (3): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True)
    (5): ReLU (inplace)
  )
  (forw1): Sequential (
    (0): PostRes (
      (conv1): Conv3d(24, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True)
      (shortcut): Sequential (
        (0): Conv3d(24, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True)
      )
    )
    (1): PostRes (
      (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (forw2): Sequential (
    (0): PostRes (
      (conv1): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (shortcut): Sequential (
        (0): Conv3d(32, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      )
    )
    (1): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (forw3): Sequential (
    (0): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
    (1): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
    (2): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (forw4): Sequential (
    (0): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
    (1): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
    (2): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (back2): Sequential (
    (0): PostRes (
      (conv1): Conv3d(131, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True)
      (shortcut): Sequential (
        (0): Conv3d(131, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True)
      )
    )
    (1): PostRes (
      (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True)
    )
    (2): PostRes (
      (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (back3): Sequential (
    (0): PostRes (
      (conv1): Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (shortcut): Sequential (
        (0): Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      )
    )
    (1): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
    (2): PostRes (
      (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
      (relu): ReLU (inplace)
      (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (maxpool1): MaxPool3d (size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (maxpool2): MaxPool3d (size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (maxpool3): MaxPool3d (size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (maxpool4): MaxPool3d (size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (unmaxpool1): MaxUnpool3d (size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
  (unmaxpool2): MaxUnpool3d (size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
  (path1): Sequential (
    (0): ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU (inplace)
  )
  (path2): Sequential (
    (0): ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU (inplace)
  )
  (drop): Dropout3d (p=0.2)
  (output): Sequential (
    (0): Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): ReLU ()
    (2): Conv3d(64, 15, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  )
)
'''


class DetectNet():
    def __init__(self):
        pass

    def createOutputNode(self, inputs):
        input = inputs[0]
        coord = inputs[1]
        # construct preblock
        preBlock_0 = Conv3D(24, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="preBlock.0")(input)
        preBlock_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="preBlock.1")(preBlock_0)
        preBlock_relu1 = Activation(K.relu)(preBlock_1)
        preBlock_3 = Conv3D(24, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="preBlock.3")(
            preBlock_relu1)
        preBlock_4 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="preBlock.4")(preBlock_3)
        preBlock_relu2 = Activation(K.relu)(preBlock_4)

        maxpool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(preBlock_relu2)

        # the first forward
        forw1_0_conv1 = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw1.0.conv1")(
            maxpool1)
        forw1_0_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw1.0.bn1")(forw1_0_conv1)
        forw1_0_relu1 = Activation(K.relu)(forw1_0_bn1)
        forw1_0_conv2 = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw1.0.conv2")(
            forw1_0_relu1)
        forw1_0_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw1.0.bn2")(forw1_0_conv2)
        # forward1 short cut
        forw1_0_shortcut_0 = Conv3D(32, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                    name="forw1.0.shortcut.0")(maxpool1)
        forw1_0_shortcut_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw1.0.shortcut.1")(
            forw1_0_shortcut_0)
        forw1_0_added = keras.layers.Add()([forw1_0_bn2, forw1_0_shortcut_1])
        forw1_0_relu = Activation(K.relu)(forw1_0_added)

        forw1_1_conv1 = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw1.1.conv1")(
            forw1_0_relu)
        forw1_1_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw1.1.bn1")(forw1_1_conv1)
        forw1_1_relu1 = Activation(K.relu)(forw1_1_bn1)
        forw1_1_conv2 = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw1.1.conv2")(
            forw1_1_relu1)
        forw1_1_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw1.1.bn2")(forw1_1_conv2)
        forw1_1_added = keras.layers.Add()([forw1_1_bn2, forw1_0_relu])
        forw1_1_relu = Activation(K.relu)(forw1_1_added)

        maxpool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(forw1_1_relu)

        # the second forward
        forw2_0_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw2.0.conv1")(
            maxpool2)
        forw2_0_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw2.0.bn1")(forw2_0_conv1)
        forw2_0_relu1 = Activation(K.relu)(forw2_0_bn1)
        forw2_0_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw2.0.conv2")(
            forw2_0_relu1)
        forw2_0_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw2.0.bn2")(forw2_0_conv2)
        # forward2 short cut
        forw2_0_shortcut_0 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                    name="forw2.0.shortcut.0")(maxpool2)
        forw2_0_shortcut_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw2.0.shortcut.1")(
            forw2_0_shortcut_0)
        forw2_0_added = keras.layers.Add()([forw2_0_bn2, forw2_0_shortcut_1])
        forw2_0_relu = Activation(K.relu)(forw2_0_added)

        forw2_1_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw2.1.conv1")(
            forw2_0_relu)
        forw2_1_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw2.1.bn1")(forw2_1_conv1)
        forw2_1_relu1 = Activation(K.relu)(forw2_1_bn1)
        forw2_1_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw2.1.conv2")(
            forw2_1_relu1)
        forw2_1_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw2.1.bn2")(forw2_1_conv2)
        forw2_1_added = keras.layers.Add()([forw2_1_bn2, forw2_0_relu])
        forw2_1_relu = Activation(K.relu)(forw2_1_added)

        maxpool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(forw2_1_relu)

        # the third forward
        forw3_0_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw3.0.conv1")(
            maxpool3)
        forw3_0_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw3.0.bn1")(forw3_0_conv1)
        forw3_0_relu1 = Activation(K.relu)(forw3_0_bn1)
        forw3_0_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw3.0.conv2")(
            forw3_0_relu1)
        forw3_0_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw3.0.bn2")(forw3_0_conv2)
        # forward3 short cut
        forw3_0_shortcut_0 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                    name="forw3.0.shortcut.0")(maxpool3)
        forw3_0_shortcut_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw3.0.shortcut.1")(
            forw3_0_shortcut_0)
        forw3_0_added = keras.layers.Add()([forw3_0_bn2, forw3_0_shortcut_1])
        forw3_0_relu = Activation(K.relu)(forw3_0_added)

        forw3_1_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw3.1.conv1")(
            forw3_0_relu)
        forw3_1_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw3.1.bn1")(forw3_1_conv1)
        forw3_1_relu1 = Activation(K.relu)(forw3_1_bn1)
        forw3_1_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw3.1.conv2")(
            forw3_1_relu1)
        forw3_1_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw3.1.bn2")(forw3_1_conv2)
        forw3_1_added = keras.layers.Add()([forw3_1_bn2, forw3_0_relu])
        forw3_1_relu = Activation(K.relu)(forw3_1_added)

        forw3_2_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw3.2.conv1")(
            forw3_1_relu)
        forw3_2_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw3.2.bn1")(forw3_2_conv1)
        forw3_2_relu1 = Activation(K.relu)(forw3_2_bn1)
        forw3_2_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw3.2.conv2")(
            forw3_2_relu1)
        forw3_2_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw3.2.bn2")(forw3_2_conv2)
        forw3_2_added = keras.layers.Add()([forw3_2_bn2, forw3_1_relu])
        forw3_2_relu = Activation(K.relu)(forw3_2_added)

        maxpool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(forw3_2_relu)

        # the fourth forward
        forw4_0_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw4.0.conv1")(
            maxpool4)
        forw4_0_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw4.0.bn1")(forw4_0_conv1)
        forw4_0_relu1 = Activation(K.relu)(forw4_0_bn1)
        forw4_0_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw4.0.conv2")(
            forw4_0_relu1)
        forw4_0_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw4.0.bn2")(forw4_0_conv2)
        # forward4 short cut
        forw4_0_shortcut_0 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                    name="forw4.0.shortcut.0")(maxpool4)
        forw4_0_shortcut_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw4.0.shortcut.1")(
            forw4_0_shortcut_0)
        forw4_0_added = keras.layers.Add()([forw4_0_bn2, forw4_0_shortcut_1])
        forw4_0_relu = Activation(K.relu)(forw4_0_added)

        forw4_1_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw4.1.conv1")(
            forw4_0_relu)
        forw4_1_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw4.1.bn1")(forw4_1_conv1)
        forw4_1_relu1 = Activation(K.relu)(forw4_1_bn1)
        forw4_1_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw4.1.conv2")(
            forw4_1_relu1)
        forw4_1_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw4.1.bn2")(forw4_1_conv2)
        forw4_1_added = keras.layers.Add()([forw4_1_bn2, forw4_0_relu])
        forw4_1_relu = Activation(K.relu)(forw4_1_added)

        forw4_2_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw4.2.conv1")(
            forw4_1_relu)
        forw4_2_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw4.2.bn1")(forw4_2_conv1)
        forw4_2_relu1 = Activation(K.relu)(forw4_2_bn1)
        forw4_2_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="forw4.2.conv2")(
            forw4_2_relu1)
        forw4_2_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="forw4.2.bn2")(forw4_2_conv2)
        forw4_2_added = keras.layers.Add()([forw4_2_bn2, forw4_1_relu])
        forw4_2_relu = Activation(K.relu)(forw4_2_added)

        # Path 1
        path1_0 = Deconvolution3D(64, kernel_size=(2, 2, 2), strides=(2, 2, 2), name="path1.0",
                                  output_shape=(None, 64, 16, 16, 16))(forw4_2_relu)
        path1_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="path1.1")(path1_0)
        path1_0_relu1 = Activation(K.relu)(path1_1)

        # comb3
        comb3 = keras.layers.concatenate([path1_0_relu1, forw3_2_relu], axis=1)

        # back3
        back3_0_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back3.0.conv1")(
            comb3)
        back3_0_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back3.0.bn1")(back3_0_conv1)
        back3_0_relu1 = Activation(K.relu)(back3_0_bn1)
        back3_0_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back3.0.conv2")(
            back3_0_relu1)
        back3_0_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back3.0.bn2")(back3_0_conv2)
        # back3 short cut
        back3_0_shortcut_0 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                    name="back3.0.shortcut.0")(comb3)
        back3_0_shortcut_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back3.0.shortcut.1")(
            back3_0_shortcut_0)
        back3_0_added = keras.layers.Add()([back3_0_bn2, back3_0_shortcut_1])
        back3_0_relu = Activation(K.relu)(back3_0_added)

        back3_1_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back3.1.conv1")(
            back3_0_relu)
        back3_1_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back3.1.bn1")(back3_1_conv1)
        back3_1_relu1 = Activation(K.relu)(back3_1_bn1)
        back3_1_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back3.1.conv2")(
            back3_1_relu1)
        back3_1_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back3.1.bn2")(back3_1_conv2)
        back3_1_added = keras.layers.Add()([back3_1_bn2, back3_0_relu])
        back3_1_relu = Activation(K.relu)(back3_1_added)

        back3_2_conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back3.2.conv1")(
            back3_1_relu)
        back3_2_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back3.2.bn1")(back3_2_conv1)
        back3_2_relu1 = Activation(K.relu)(back3_2_bn1)
        back3_2_conv2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back3.2.conv2")(
            back3_2_relu1)
        back3_2_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back3.2.bn2")(back3_2_conv2)
        back3_2_added = keras.layers.Add()([back3_2_bn2, back3_1_relu])
        back3_2_relu = Activation(K.relu)(back3_2_added)

        # Path 2
        path2_0 = Deconvolution3D(64, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="path2.0",
                                  output_shape=(None, 64, 32, 32, 32))(back3_2_relu)
        path2_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="path2.1")(path2_0)
        path2_0_relu1 = Activation(K.relu)(path2_1)

        # comb2
        comb2 = keras.layers.concatenate([path2_0_relu1, forw2_1_relu, coord], axis=1)

        # back 2
        back2_0_conv1 = Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back2.0.conv1")(
            comb2)
        back2_0_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back2.0.bn1")(back2_0_conv1)
        back2_0_relu1 = Activation(K.relu)(back2_0_bn1)
        back2_0_conv2 = Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back2.0.conv2")(
            back2_0_relu1)
        back2_0_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back2.0.bn2")(back2_0_conv2)
        # back2 short cut
        back2_0_shortcut_0 = Conv3D(128, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                                    name="back2.0.shortcut.0")(comb2)
        back2_0_shortcut_1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back2.0.shortcut.1")(
            back2_0_shortcut_0)
        back2_0_added = keras.layers.Add()([back2_0_bn2, back2_0_shortcut_1])
        back2_0_relu = Activation(K.relu)(back2_0_added)

        back2_1_conv1 = Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back2.1.conv1")(
            back2_0_relu)
        back2_1_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back2.1.bn1")(back2_1_conv1)
        back2_1_relu1 = Activation(K.relu)(back2_1_bn1)
        back2_1_conv2 = Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back2.1.conv2")(
            back2_1_relu1)
        back2_1_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back2.1.bn2")(back2_1_conv2)
        back2_1_added = keras.layers.Add()([back2_1_bn2, back2_0_relu])
        back2_1_relu = Activation(K.relu)(back2_1_added)

        back2_2_conv1 = Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back2.2.conv1")(
            back2_1_relu)
        back2_2_bn1 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back2.2.bn1")(back2_2_conv1)
        back2_2_relu1 = Activation(K.relu)(back2_2_bn1)
        back2_2_conv2 = Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", name="back2.2.conv2")(
            back2_2_relu1)
        back2_2_bn2 = BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, name="back2.2.bn2")(back2_2_conv2)

        back2_2_added = keras.layers.Add()([back2_2_bn2, back2_1_relu])
        feat = Activation(K.relu)(back2_2_added)
        dropout_2 = Dropout(0.2)(feat)

        # Output
        output_0 = Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid", name="output.0")(dropout_2)
        output_relu = Activation(K.relu)(output_0)
        output_2 = Conv3D(5 * len(config['anchors']), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid",
                          name="output.2")(output_relu)
        print(feat.shape)
        print(output_2.shape)
        size = K.int_shape(output_2)
        output = Lambda(self.postprocessOutputTensor,
                        output_shape=(-1, size[2], size[3], size[4], len(config['anchors']), 5))(output_2)
        return output, feat

    def postprocessOutputTensor(self, output):
        print("I am here")
        size = K.int_shape(output)
        print (size)
        out = K.reshape(output, (-1, size[1], size[2] * size[3] * size[4]))
        print(out.shape)
        out = K.permute_dimensions(out, (0, 2, 1))
        print(out.shape)
        out = K.reshape(out, (-1, size[2], size[3], size[4], len(config['anchors']), 5))
        print(out.shape)
        return out