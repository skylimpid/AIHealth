import os
import pandas
import time
import numpy as np

from Utils.data_set import DataSet
from Utils.data_set_utils import simpleCrop, sample, ClassifierDataAugment
from Training.configuration_training import cfg
from Net.tensorflow_model.classifier_net import get_config


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
        self.candidate_box = {}
        for f in os.listdir(cfg.DIR.classifier_net_intermediate_candidate_box):
            name = f.split('_')[0]
            self.candidate_box[name] = os.path.join(cfg.DIR.classifier_net_intermediate_candidate_box, f)

        self.pbb_label = {}
        for f in os.listdir(cfg.DIR.classifier_net_intermediate_pbb_label):
            name = f.split('_')[0]
            self.pbb_label[name] = os.path.join(cfg.DIR.classifier_net_intermediate_pbb_label, f)
        #print(self.pbb_label)
        idcs = np.load(split)

        # print(idcs)
        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx) for idx in idcs]
        # print(self.filenames)
        if phase != 'test':
            self.yset = {}
            labels = np.array(pandas.read_csv(labelfile))
            for i in range(len(labels)):
                self.yset[labels[i][0]] = labels[i][1]
        idcs = [f.split('-')[0] for f in idcs]
        # print (idcs)

        self.crop = simpleCrop(config, phase)
        self.index = 0
        self.length = self.__len__()

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        img = np.load(self.filenames[idx])
        fileName = self.filenames[idx].split('/')[-1].split('_')[0]
        print(self.candidate_box[fileName])
        pbb = np.load(self.candidate_box[fileName])
        print("shape1", pbb.shape)
        pbb = np.squeeze(pbb)
        print("shape2", pbb.shape)
        pbb_label = np.load(self.pbb_label[fileName])
        pbb_label = np.squeeze(pbb_label)
        #print(pbb_label.shape)
        conf_list = pbb[:, 0]
        T = self.T
        topk = self.topk
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
            #print(id)
            target = pbb[id, 1:]
            #print(target)
            isnod = pbb_label[id]
            #print(isnod)
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
            y = np.array([self.yset[fileName]])
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


if __name__ == "__main__":

    dataset = TrainingClassifierData(cfg.DIR.preprocess_result_path, cfg.DIR.bbox_path,
                                     cfg.DIR.kaggle_full_labels,
                                     cfg.DIR.classifier_net_train_data_path
                                     , get_config())
    dataset.__getitem__(0)