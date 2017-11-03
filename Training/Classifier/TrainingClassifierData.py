import os
import pandas
import time
import numpy as np

from Utils.DataSet import DataSet
from Utils.nms_cython import nms, iou
from Utils.DataSetUtils import simpleCrop, sample, ClassifierDataAugment



class TrainingClassifierData(DataSet):

    def __init__(self, preprocess_result_dir, bboxpath_dir, labelfile, split, config,  phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')

        self.random_sample = config['random_sample']
        self.T = config['T']
        self.topk = config['topk']
        self.crop_size = config['crop_size']
        self.stride = config['stride']
        self.augtype = config['augtype']
        # self.labels = np.array(pandas.read_csv(config['labelfile']))

        datadir = preprocess_result_dir
        bboxpath = bboxpath_dir
        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []

        idcs = np.load(split)

        # print(idcs)
        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx) for idx in idcs]
        # print(self.filenames)
        if phase != 'test':
            labels = np.array(pandas.read_csv(labelfile))
            self.yset = np.array([labels[labels[:, 0] == f.split('-')[0].split('_')[0], 1] for f in idcs]).astype(
                'int')
        idcs = [f.split('-')[0] for f in idcs]
        # print (idcs)

        for idx in idcs:
            print(idx)
            pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
            pbb = pbb[pbb[:, 0] > config['conf_th']]
            pbb = nms(pbb, config['nms_th'], self.topk*100)
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
                #             if idx.startswith()
            self.candidate_box.append(pbb)
            self.pbb_label.append(np.array(pbb_label))
        self.crop = simpleCrop(config, phase)
        self.index = 0
        self.length = self.__len__()

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
        coordlist = np.zeros([topk, 3, int(self.crop_size[0] / self.stride), int(self.crop_size[1] / self.stride),
                              int(self.crop_size[2] / self.stride)]).astype('float32')
        padmask = np.concatenate([np.ones(len(chosenid)), np.zeros(self.topk - len(chosenid))])
        isnodlist = np.zeros([topk])

        for i, id in enumerate(chosenid):
            target = pbb[id, 1:]
            isnod = pbb_label[id]
            crop, coord = self.crop(img, target)
            if self.phase == 'train':
                crop, coord = ClassifierDataAugment(crop, coord,
                                      ifflip=self.augtype['flip'], ifrotate=self.augtype['rotate'],
                                      ifswap=self.augtype['swap'])
            crop = crop.astype(np.float32)
            croplist[i] = crop
            coordlist[i] = coord
            isnodlist[i] = isnod

        if self.phase != 'test':
            y = np.array([self.yset[idx]])
            return croplist, coordlist, isnodlist, y
        else:
            return croplist, coordlist, isnodlist

    def __len__(self):
        if self.phase != 'test':
            return len(self.candidate_box)
        else:
            return len(self.candidate_box)

    def getNextBatch(self, batch_size):
        final_cropList = None
        final_coordlist = None
        final_isnodlist = None
        final_y = None

        if self.phase != 'test':
            for i in range(batch_size):
                if self.index >= self.length:
                    break
                croplist, coordlist, isnodlist, y = self.__getitem__(self.index)

                if final_cropList is None:
                    final_cropList = np.expand_dims(croplist, axis=0)
                else:
                    final_cropList = np.append(final_cropList, np.expand_dims(croplist, axis=0), axis=0)

                if final_coordlist is None:
                    final_coordlist = np.expand_dims(coordlist, axis=0)
                else:
                    final_coordlist = np.append(final_coordlist, np.expand_dims(coordlist, axis=0), axis=0)

                if final_isnodlist is None:
                    final_isnodlist = np.expand_dims(isnodlist, axis=0)
                else:
                    final_isnodlist = np.append(final_isnodlist, np.expand_dims(isnodlist, axis=0), axis=0)

                if final_y is None:
                    #final_y = np.expand_dims(y, axis=0)
                    final_y = y
                else:
                    #final_y = np.append(final_y, np.expand_dims(y, axis=0), axis=0)
                    final_y = np.append(final_y, y, axis=0)

                self.index = self.index + 1
            return final_cropList, final_coordlist, final_isnodlist, final_y
        else:
            for i in range(batch_size):
                if self.index >= self.length:
                    break
                croplist, coordlist, isnodlist = self.__getitem__(self.index)

                if final_cropList is None:
                    final_cropList = np.expand_dims(croplist, axis=0)
                else:
                    final_cropList = np.append(final_cropList, np.expand_dims(croplist, axis=0), axis=0)

                if final_coordlist is None:
                    final_coordlist = np.expand_dims(coordlist, axis=0)
                else:
                    final_coordlist = np.append(final_coordlist, np.expand_dims(coordlist, axis=0), axis=0)

                if final_isnodlist is None:
                    final_isnodlist = np.expand_dims(isnodlist, axis=0)
                else:
                    final_isnodlist = np.append(final_isnodlist, np.expand_dims(isnodlist, axis=0), axis=0)

                self.index = self.index + 1
            return final_cropList, final_coordlist, final_isnodlist

    def hasNextBatch(self):
        return self.index < self.length

    def reset(self):
        self.index = 0
