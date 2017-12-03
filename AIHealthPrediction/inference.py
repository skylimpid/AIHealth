import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Training.constants import DIMEN_X, DIMEN_Y
from Preprocessing.preprocessing_dicom import dicom_python
from Preprocessing.preprocessing import process_mask, resample, lumTrans
import numpy as np
from Utils.split_combine import SplitComb
from Training.constants import SIDE_LEN, MARGIN
from AIHealthPrediction.inference_config import config
from AIHealthPrediction.inference_data import InferenceDataSet
import tensorflow as tf
import time
from Net.tensorflow_model.detector_net import DetectorNet, get_model as detect_net_model
from Net.tensorflow_model.classifier_net import get_model as classifier_net_model
from Utils.utils import nms
from AIHealthPrediction.html_generator import generate_html_report



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
    return x

parser = argparse.ArgumentParser(description='AIHealth Inference')
parser.add_argument('--dicom_dir', dest='dicom_dir', required=True, type=dicom_dir_type)
parser.add_argument('--confidence_level', dest='confidence_level', type=confidence_level_type, default=0.5)
parser.add_argument('--patient_id', dest='patient_id', type=str, default=None)

def inference():
    args = parser.parse_args()
    dicom_dir = args.dicom_dir
    confidence_level = args.confidence_level
    patient_id = args.patient_id
    if patient_id is not None:
        inference_each(patient_id=patient_id, dicom_dir=dicom_dir, confidence_level=confidence_level)
    else:
        patients = os.listdir(dicom_dir)
        for patient in patients:
            inference_each(patient_id=patient, dicom_dir=dicom_dir, confidence_level=confidence_level)


def inference_each(patient_id, dicom_dir, confidence_level):

    init_start = time.time()
    patient_dicom = os.path.join(dicom_dir, patient_id)

    if not os.path.isdir(patient_dicom):
        print("Bad data. {} should be a directory.".format(patient_dicom))

    if not os.path.exists(patient_dicom):
        print("The dicom file of the patient:{} does not exist in this directory:{}"
                         .format(patient_id, dicom_dir))
        exit(-1)

    if (len(os.listdir(patient_dicom)) == 0):
        print("There are no dicom files for patient:{} under directory:{}".format(patient_id, patient_dicom))
        exit(-1)

    print("Start to work on patient:{}'s data. ".format(patient_id))

    print("Preprocessing the dicom files. The total dicom files for this patient:{} are:{}".format(patient_id,
        len(os.listdir(patient_dicom))))
    start = time.time()
    resolution = np.array([1, 1, 1])
    im, m1, m2, spacing = dicom_python(patient_dicom)
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

    print("Predict....")
    start = time.time()
    # prepare the dataset for detectnet and classifiernet
    split_combine = SplitComb(side_len=SIDE_LEN, max_stride=config['max_stride'],
                              stride=config['stride'], margin=MARGIN,
                              pad_value=config['pad_value'])

    data_set = InferenceDataSet(config = config, split_combiner = split_combine,
                                preprocessed_clean_img = cleaned_dicom_files)

    bbox = detector_net_predict(data_set, split_combine)

    # filter by given confidence_level
    bbox[:, 0] = sigmoid(bbox[:, 0])
    bbox = bbox[bbox[:, 0] >= confidence_level]

    predict = classifier_net_predict(data_set, bbox)

    end = time.time()
    print("Get the predict results. The total time spent:{}".format(end-start))

    print("Generating the diagnosis report.")
    generate_html_report(report_dir=config['report_dir'], patient_id=patient_id, clean_img=cleaned_dicom_files,
                         bbox=bbox, predict=predict)
    print("The Diagnosis report has been generated for the patient:{}.".format(patient_id))

    end = time.time()
    print("The total time spend for patient:{} is {}".format(patient_id, str(end-init_start)))


def detector_net_predict(data_set, split_combine):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 1, DIMEN_X, DIMEN_X, DIMEN_X])
    coord = tf.placeholder(tf.float32, shape=[None, 3, DIMEN_Y, DIMEN_Y, DIMEN_Y])

    _, detector_net_object, _, pbb = detect_net_model()

    _, _, _, _, _, _, _, _, feat, out = detector_net_object.getDetectorNet(X, coord)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
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
        #print(final_out.shape)
        output = split_combine.combine(final_out, nzhw=nzhw)
        thresh = -3
        bbox, mask = pbb(output, thresh, ismask=True)

        bbox = bbox[bbox[:, 0] >= config['conf_th']]
        bbox = nms(bbox, config['nms_th'])
        return bbox

def classifier_net_predict(data_set, bbox):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, config['topk'], 1, DIMEN_X, DIMEN_X, DIMEN_X])
    coord = tf.placeholder(tf.float32, shape=[None, config['topk'], 3, DIMEN_Y, DIMEN_Y, DIMEN_Y])
    detector_net = DetectorNet()
    _, classifier_net_object = classifier_net_model(detector_net)
    _, case_pred, _, _, _ = classifier_net_object.get_classifier_net(X, coord)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        # load the previous trained detector_net model
        value_list = []
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/detector_scope'))
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/cl_scope'))
        saver = tf.train.Saver(value_list)
        saver.restore(sess, tf.train.latest_checkpoint(config['classifier_net_ckg']))

        imgs, coord2 = data_set.getClassiferDataSet(bbox=bbox)

        case_pred_out = sess.run(case_pred, feed_dict={X:imgs, coord:coord2})
        return case_pred_out[0][0]





if __name__ == '__main__':
    inference()

    """
    # prepare the dataset for detectnet and classifiernet
    cleaned_dicom_files = np.load('/home/xuan/AIHealthData/preprocess_result/fb57fc6377fd37bb5d42756c2736586c_clean.npy')
    split_combine = SplitComb(side_len=SIDE_LEN, max_stride=config['max_stride'],
                              stride=config['stride'], margin=MARGIN,
                              pad_value=config['pad_value'])

    data_set = InferenceDataSet(config=config, split_combiner=split_combine,
                                preprocessed_clean_img=cleaned_dicom_files)

    bbox = np.load('/home/xuan/AIHealthData/classifier_intermediate/candidate_box/fb57fc6377fd37bb5d42756c2736586c_candidate.npy')

    imgs, coord2, _ = data_set.getDetectorDataSet()
    print(imgs.shape)
    print(coord2.shape)

    cropList, coord_list = data_set.getClassiferDataSet(bbox)
    print(cropList.shape)
    print(coord_list.shape)

    detector_net_predict(data_set, split_combine)
    classifier_net_predict(data_set, bbox)
    """

