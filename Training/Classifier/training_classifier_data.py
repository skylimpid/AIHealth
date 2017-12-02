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

        datadir = preprocess_result_dir
        # bboxpath = bboxpath_dir
        self.phase = phase
        self.candidate_box = {}
        for f in os.listdir(cfg.DIR.classifier_net_intermediate_candidate_box):
            name = f.split('_')[0]
            self.candidate_box[name] = os.path.join(cfg.DIR.classifier_net_intermediate_candidate_box, f)

        self.pbb_label = {}
        for f in os.listdir(cfg.DIR.classifier_net_intermediate_pbb_label):
            name = f.split('_')[0]
            self.pbb_label[name] = os.path.join(cfg.DIR.classifier_net_intermediate_pbb_label, f)

        self.idcs = np.load(split)

        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx) for idx in self.idcs]

        # we should have labelfile all the time
        self.yset = {}
        labels = np.array(pandas.read_csv(labelfile))
        for i in range(len(labels)):
            self.yset[labels[i][0]] = labels[i][1]

        self.crop = simpleCrop(config, phase)
        self.index = 0
        self.length = self.__len__()


    def __getitem__(self, idx, split=None, test=False):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        img = np.load(self.filenames[idx])
        file_name = self.filenames[idx].split('/')[-1].split('_')[0]

        pbb = np.load(self.candidate_box[file_name])
        pbb_label = np.load(self.pbb_label[file_name])

        if test:
            print(self.filenames[idx])
            print(file_name)
            print(self.candidate_box[file_name])
            print("pbb shape: {}, ndim: {}, len: {}, pbb_label shape: {}".format(pbb.shape, pbb.ndim, len(pbb), pbb_label.shape))

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
        #padmask = np.concatenate([np.ones(len(chosenid)), np.zeros(self.topk - len(chosenid))])
        isnodlist = np.zeros([topk])

        for i, id in enumerate(chosenid):
            target = pbb[id, 1:]
            isnod = pbb_label[id]
            crop, coord = self.crop(img, target)
            if self.phase == 'train':
                crop, coord = ClassifierDataAugment(crop, coord,
                                      ifflip=self.augtype['flip'], ifrotate=self.augtype['rotate'],
                                      ifswap=self.augtype['swap'])

            crop = (crop.astype(np.float32)-128)/128
            croplist[i] = crop
            coordlist[i] = coord
            isnodlist[i] = isnod

        label = np.array([self.yset[file_name]])
        return croplist, coordlist, isnodlist, label, file_name


    def getNextBatch(self, batch_size):
        final_cropList = None
        final_coordlist = None
        final_isnodlist = None
        final_labels = None
        final_file_names = None

        for i in range(batch_size):

            if not self.hasNextBatch():
                break

            croplist, coordlist, isnodlist, label, file_name = self.__getitem__(self.index)

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

            if final_labels is None:
                final_labels = label
            else:
                final_labels = np.append(final_labels, label, axis=0)

            if final_file_names is None:
                final_file_names = np.array(file_name)
            else:
                final_file_names = np.append(final_file_names, file_name)

            self.index += 1

        # labels
        final_labels = final_labels.reshape((-1, 1))
        return final_cropList, final_coordlist, final_isnodlist, final_labels, final_file_names


    def __len__(self):
            return len(self.idcs)

    def hasNextBatch(self):
        return self.index < self.length

    def reset(self):
        self.index = 0


if __name__ == "__main__":

    dataset = TrainingClassifierData(cfg.DIR.preprocess_result_path, cfg.DIR.bbox_path,
                                     cfg.DIR.kaggle_full_labels,
                                     cfg.DIR.classifier_net_train_data_path
                                     , get_config(), phase = "train")

    a, b, c, d, f = dataset.__getitem__(108, test=True)

    print(a)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(f)

    _, _, _, labels, file_names = dataset.getNextBatch(5)
    print(labels)
    print(file_names)