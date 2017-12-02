import argparse
import os
from Preprocessing.preprocessing_dicom import dicom_python
from Preprocessing.preprocessing import process_mask, resample, lumTrans
import numpy as np
from Utils.split_combine import SplitComb
from Training.constants import SIDE_LEN, MARGIN
from AIHealthPrediction.inference_config import config
from AIHealthPrediction.inference_data import InferenceDataSet
import tensorflow as tf
import time
from Training.constants import DIMEN_X, DIMEN_Y
from Net.tensorflow_model.detector_net import get_model as detect_net_model
from Utils.utils import nms

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def confidence_level_type(x):
    x = float(x)
    if x < 0:
        raise argparse.ArgumentTypeError("Minumum confidence_level is 0")
    return x

def dicom_dir_type(x):
    x = str(x)
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("The specified dicom_dir:{} does not exist.".format(x))
    if len([name for name in os.listdir(x) if os.path.isfile(os.path.join(x, name))]) <= 0:
        raise argparse.ArgumentTypeError("Can not find any dicom files under dicom_dir:{}.".format(x))
    return x

parser = argparse.ArgumentParser(description='AIHealth Inference')
parser.add_argument('--dicom_dir', dest='dicom_dir', required=True, type=dicom_dir_type)
parser.add_argument('--confidence_level', dest='confidence_level', type=confidence_level_type, default=0.5)

def inference():
    args = parser.parse_args()
    dicom_dir = args.dicom_dir
    confidence_level = args.confidence_level

    print("Start to preprocess the dicom files. The total dicom files for this patient are:{}".format(
        len(os.listdir(dicom_dir))))
    start = time.time()
    resolution = np.array([1, 1, 1])
    im, m1, m2, spacing = dicom_python(dicom_dir)
    Mask = m1 + m2

    newshape = np.round(np.array(Mask.shape) * spacing / resolution)
    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T
    extendbox = extendbox.astype('int')

    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1 + dm2
    Mask = m1 + m2
    extramask = dilatedMask - Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)] = -2000
    sliceim = lumTrans(im)
    sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
    bones = sliceim * extramask > bone_thresh
    sliceim[bones] = pad_value
    sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
    sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
               extendbox[1, 0]:extendbox[1, 1],
               extendbox[2, 0]:extendbox[2, 1]]
    cleaned_dicom_files = sliceim2[np.newaxis, ...]
    print(cleaned_dicom_files.shape)
    end = time.time()
    print("Finish the preprocessing. The total time spent:{}".format(end-start))

    # prepare the dataset for detectnet and classifiernet
    split_combine = SplitComb(side_len=SIDE_LEN, max_stride=config['max_stride'],
                              stride=config['stride'], margin=MARGIN,
                              pad_value=config['pad_value'])

    data_set = InferenceDataSet(config = config, split_combiner = split_combine,
                                preprocessed_clean_img = cleaned_dicom_files)

    bbox = detector_net_predict(data_set, split_combine, confidence_level)


def detector_net_predict(data_set, split_combine, confidence_level):
    X = tf.placeholder(tf.float32, shape=[None, 1, DIMEN_X, DIMEN_X, DIMEN_X])
    coord = tf.placeholder(tf.float32, shape=[None, 3, DIMEN_Y, DIMEN_Y, DIMEN_Y])

    _, detector_net_object, _, pbb = detect_net_model()

    feat, out = detector_net_object.getDetectorNet(X, coord)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # load the previous trained detector_net model
        value_list = []
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope'))
        saver = tf.train.Saver(value_list)
        saver.restore(sess, tf.train.latest_checkpoint(config['detector_net_ckg']))

        imgs, coord2, nzhw = data_set.getDetectorDataSet()

        total_size_per_img = imgs.shape[0]

        index = 0
        final_out = None

        start_time = time.time()
        batch_size = config['detector_net_batch_size']
        while index + batch_size < total_size_per_img:
            feat_predict, out_predict = sess.run([feat, out], feed_dict={
                X: imgs[index:index + batch_size],
                coord: coord2[index:index + batch_size]})
            if final_out is None:
                final_out = out_predict
            else:
                final_out = np.concatenate((final_out, out_predict), axis=0)

            index = index + batch_size

        if index < total_size_per_img:
            feat_predict, out_predict = sess.run([feat, out], feed_dict={
                X: imgs[index:], coord: coord2[index:]})
            if final_out is None:
                final_out = out_predict
            else:
                final_out = np.concatenate((final_out, out_predict), axis=0)
        print(final_out.shape)
        end_time = time.time()
        print("The time spent in detector_net is: {}".format(end_time - start_time))
        output = split_combine.combine(final_out, nzhw=nzhw)
        thresh = -3
        bbox, mask = pbb(output, thresh, ismask=True)

    bbox[:, 0] = sigmoid(bbox[:, 0])
    bbox = bbox[bbox[:, 0] >= confidence_level]
    bbox = nms(bbox, config['nms_th'])
    return bbox

if __name__ == '__main__':
    inference()