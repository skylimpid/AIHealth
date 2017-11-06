import time
import os
import math

import numpy as np
from Utils.DataSet import DataSet
from Utils.DataSetUtils import Crop, LabelMapping, DetectorDataAugmentRotate, DetectorDataAugmentFlip


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
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        self.r_rand = config['r_rand_crop']
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

        # Keep the original copy
        self.sample_bboxes = labels

        ##
        ## Filter bounding boxes, and also increase the presence for big ones
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:
                    for t in l:
                        #t[3] -> diameter of the curated bounding box
                        if t[3] > sizelim:
                            self.bboxes.append([np.concatenate([[i], t])])
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[i], t])]] * 2
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[i], t])]] * 4
            self.bboxes = np.concatenate(self.bboxes, axis=0)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)
        self.index = 0
        self.length = self.__len__()
        self.sample_pool = None
        self.label_pool = None
        self.coord_pool = None

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        isRandomImg = False
        ## If we've reached the last of bboxes, will start random cropping for negative sampling
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
                ## bboxes from the same patient
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase == 'train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                ## sample: the cropped image part around the bounding box
                ## target: the bbox[1:] in the new bounding box coordinate
                ## bboxes: dito
                ## coord: a normalized meshgrid

                label = self.label_mapping(sample.shape[1:], target, bboxes)
                sample_total = np.expand_dims(sample, axis=0)
                coord_total = np.expand_dims(coord, axis=0)
                label_total = np.expand_dims(label, axis=0)
                if self.phase == 'train' and not isRandom:
                    if self.augtype['rotate']:
                        sample2, target2, bboxes2, coord2, success2 = DetectorDataAugmentRotate(sample, target,
                                                                                                bboxes, coord)
                        if success2:
                            sample_total = np.concatenate((sample_total, np.expand_dims(sample2, axis=0)), axis=0)
                            coord_total = np.concatenate((coord_total, np.expand_dims(coord2, axis=0)), axis=0)
                            label_local = self.label_mapping(sample2.shape[1:], target2, bboxes2)
                            label_total = np.concatenate((label_total, np.expand_dims(label_local, axis=0)), axis=0)

                    if self.augtype['swap']:
                        sample2, target2, bboxes2, coord2, success2 = DetectorDataAugmentRotate(sample, target,
                                                                                                bboxes, coord)
                        if success2:
                            sample_total = np.concatenate((sample_total, np.expand_dims(sample2, axis=0)), axis=0)
                            coord_total = np.concatenate((coord_total, np.expand_dims(coord2, axis=0)), axis=0)
                            label_local = self.label_mapping(sample2.shape[1:], target2, bboxes2)
                            label_total = np.concatenate((label_total, np.expand_dims(label_local, axis=0)), axis=0)

                    if self.augtype['flip']:
                        sample2, target2, bboxes2, coord2, success2 = DetectorDataAugmentFlip(sample, target,
                                                                                              bboxes, coord)
                        if success2:
                            sample_total = np.concatenate((sample_total, np.expand_dims(sample2, axis=0)), axis=0)
                            coord_total = np.concatenate((coord_total, np.expand_dims(coord2, axis=0)), axis=0)
                            label_local = self.label_mapping(sample2.shape[1:], target2, bboxes2)
                            label_total = np.concatenate((label_total, np.expand_dims(label_local, axis=0)), axis=0)

                sample = np.copy(sample_total)
                coord = np.copy(coord_total)
                label = np.copy(label_total)
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)
                label = self.label_mapping(sample.shape[1:], target, bboxes)
                sample = np.expand_dims(sample, axis=0)
                label = np.expand_dims(label, axis=0)
                coord = np.expand_dims(coord, axis=0)

            sample = (sample.astype(np.float32) - 128) / 128
            # if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            return sample, label, coord
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
            return math.ceil(len(self.bboxes)/(1 - self.r_rand))
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)

    def getNextBatch(self, batch_size):
        assert(self.phase != 'test')

        while (self.sample_pool is None or len(self.sample_pool) < batch_size) and self.index < self.length:

            sample, label, coord = self.__getitem__(self.index)
            if self.sample_pool is None:
                self.sample_pool = np.copy(sample)
                self.label_pool = np.copy(label)
                self.coord_pool = np.copy(coord)
            else:
                try:
                    self.sample_pool = np.concatenate((self.sample_pool, sample), axis=0)
                    self.label_pool = np.concatenate((self.label_pool, label), axis=0)
                    self.coord_pool = np.concatenate((self.coord_pool, coord), axis=0)
                except ValueError:
                    continue
            self.index += 1

        if len(self.sample_pool) >= batch_size:
            samples = np.copy(self.sample_pool[0:batch_size])
            labels_out = np.copy(self.label_pool[0:batch_size])
            coords_out = np.copy(self.coord_pool[0:batch_size])
            self.sample_pool = self.sample_pool[batch_size:]
            self.label_pool = self.label_pool[batch_size:]
            self.coord_pool = self.coord_pool[batch_size:]
            return samples, labels_out, coords_out
        else:
            samples = np.copy(self.sample_pool)
            labels_out = np.copy(self.label_pool)
            coords_out = np.copy(self.coord_pool)
            self.sample_pool = None
            self.label_pool = None
            self.coord_pool= None
            return samples, labels_out, coords_out

    def hasNextBatch(self):
        return self.index < self.length or (self.sample_pool is not None and len(self.sample_pool) > 0)

    def reset(self):
        self.index = 0


if __name__ == "__main__":
    from Net.tensorflow_model.DetectorNet import get_model

    config, net, loss, get_pbb = get_model()
    from Training.configuration_training import cfg
    data_set = TrainingDetectorData(cfg.DIR.preprocess_result_path,
                                    cfg.DIR.detector_net_train_data_path,
                                    config,
                                    phase='train')

    data, labels, coords = data_set.__getitem__(0)
    print(data.shape)
    print(labels.shape)
    print(coords.shape)
    """
    batch_data, batch_labels, batch_coord = data_set.getNextBatch(500)
    print(batch_labels.shape)
    print(batch_data.shape)
    print(batch_coord.shape)

    print(data_set.hasNextBatch())
    """
