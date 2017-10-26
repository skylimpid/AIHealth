import numpy as np
from Utils.DataSet import DataSet
from Utils.DataSetUtils import Crop, DetectorDataAugment, LabelMapping
import time
import os


class TrainingDetectorData(DataSet):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        idcs = np.load(split_path)
        if phase != 'test':
            idcs = [f for f in idcs if (f not in self.blacklist)]
        #print("test:{}".format(idcs))

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]

        labels = []

        for idx in idcs:
            l = np.load(os.path.join(data_dir, '%s_label.npy' % idx))
            if np.all(l == 0):
                l = np.array([])
            labels.append(l)
        #print("labels:{}".format(labels))
        self.sample_bboxes = labels
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:
                    for t in l:
                        if t[3] > sizelim:
                            self.bboxes.append([np.concatenate([[i], t])])
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[i], t])]] * 2
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[i], t])]] * 4
            self.bboxes = np.concatenate(self.bboxes, axis=0)
        #print("bboxes:{}".format(self.bboxes))
        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)
        self.index = 0
        self.length = self.__len__()

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        isRandomImg = False
        if self.phase != 'test':
            if idx >= len(self.bboxes):
                isRandom = True
                idx = idx % len(self.bboxes)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False

        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase == 'train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                if self.phase == 'train' and not isRandom:
                    sample, target, bboxes, coord = DetectorDataAugment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype['flip'],
                                                            ifrotate=self.augtype['rotate'],
                                                            ifswap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)
            label = self.label_mapping(sample.shape[1:], target, bboxes)
            sample = (sample.astype(np.float32) - 128) / 128
            # if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            return sample.tolist(), label.tolist(), coord.tolist()
        else:
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                          constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=int(self.split_comber.side_len / self.stride),
                                                    max_stride=int(self.split_comber.max_stride / self.stride),
                                                    margin=int(self.split_comber.margin / self.stride))
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            return imgs, bboxes, coord2, np.array(nzhw), self.filenames[idx]

    def __len__(self):
        if self.phase == 'train':
            return len(self.bboxes) / (1 - self.r_rand)
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)

    def getNextBatch(self, batch_size):
        samples=[]
        labels=[]
        coords=[]
        nzhws=[]
        if self.phase != 'test':
            for i in range(batch_size):
                sample, label, coord = self.__getitem__(self.index)
                samples.append(sample)
                labels.append(label)
                coords.append(coord)
                self.index = self.index + 1
            return samples, labels, coords
        else:
            for i in range(batch_size):
                sample, label, coord, nzhw = self.__getitem__(self.index)
                samples.append(sample)
                labels.append(label)
                coords.append(coord)
                nzhws.append(nzhw)
                self.index = self.index + 1
            return samples, labels, coords, nzhws

    def hasNextBatch(self):
        return self.index < self.length

    def reset(self):
        self.index = 0
