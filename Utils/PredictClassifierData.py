import os
import time
import numpy as np
import tensorflow as tf
from Utils.utils import nms, iou
from Utils.DataSet import DataSet
from Utils.DataSetUtils import simpleCrop, sample, augment


class PredictClassifierData(DataSet):
    def __init__(self, split, config, phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')

        self.random_sample = config['random_sample']
        self.T = config['T']
        self.topk = config['topk']
        self.crop_size = config['crop_size']
        self.stride = config['stride']
        self.augtype = config['augtype']
        self.filling_value = config['filling_value']

        # self.labels = np.array(pandas.read_csv(config['labelfile']))

        datadir = config['datadir']
        bboxpath = config['bboxpath']
        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []

        idcs = split
        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx.split('-')[0]) for idx in idcs]
        if self.phase != 'test':
            self.yset = 1 - np.array([f.split('-')[1][2] for f in idcs]).astype('int')

        for idx in idcs:
            pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
            pbb = pbb[pbb[:, 0] > config['conf_th']]
            pbb = nms(pbb, config['nms_th'])

            lbb = np.load(os.path.join(bboxpath, idx + '_lbb.npy'))
            pbb_label = []

            for p in pbb:
                isnod = False
                for l in lbb:
                    score = iou(p[1:5], l)
                    if score > config['detect_th']:
                        isnod = True
                        break
                pbb_label.append(isnod)
            # if idx.startswith()
            self.candidate_box.append(pbb)
            self.pbb_label.append(np.array(pbb_label))
        self.crop = simpleCrop(config, phase)


    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        pbb = self.candidate_box[idx]
        pbb_label = self.pbb_label[idx]
        conf_list = pbb[:, 0]
        T = self.T
        topk = self.topk
        img = np.load(self.filenames[idx])
        if self.random_sample and self.phase == 'train':
            chosenid = sample(conf_list, topk, T=T)
            # chosenid = conf_list.argsort()[::-1][:topk]
        else:
            chosenid = conf_list.argsort()[::-1][:topk]
        croplist = np.zeros([topk, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]]).astype('float32')
        coordlist = np.zeros([topk, 3, self.crop_size[0] / self.stride, self.crop_size[1] / self.stride,
                              self.crop_size[2] / self.stride]).astype('float32')
        padmask = np.concatenate([np.ones(len(chosenid)), np.zeros(self.topk - len(chosenid))])
        isnodlist = np.zeros([topk])

        for i, id in enumerate(chosenid):
            target = pbb[id, 1:]
            isnod = pbb_label[id]
            crop, coord = self.crop(img, target)
            if self.phase == 'train':
                crop, coord = augment(crop, coord,
                                      ifflip=self.augtype['flip'], ifrotate=self.augtype['rotate'],
                                      ifswap=self.augtype['swap'], filling_value=self.filling_value)
            crop = crop.astype(np.float32)
            croplist[i] = crop
            coordlist[i] = coord
            isnodlist[i] = isnod

        if self.phase != 'test':
            y = np.array([self.yset[idx]])
            return tf.stack(croplist.astype(float)), tf.stack(coordlist.astype(float)),\
                   tf.stack(isnodlist.astype(int)), tf.stack(y)
        else:
            return tf.stack(croplist.astype(float)), tf.stack(coordlist.astype(float))

    def __len__(self):
        if self.phase != 'test':
            return len(self.candidate_box)
        else:
            return len(self.candidate_box)