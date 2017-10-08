import numpy as np
from Utils.DataSet import DataSet
from Utils.DataSetUtils import augment, Crop, LabelMapping
import os
import time
import tensorflow as tf

class PredictDetectorData(DataSet):
    def __init__(self, split, config, phase='train',split_comb=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']       
        self.stride = config['stride']       
        sizelim = config['sizelim']/config['reso']
        sizelim2 = config['sizelim2']/config['reso']
        sizelim3 = config['sizelim3']/config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        data_dir = config['datadir']
        self.pad_value = config['pad_value']
        
        self.split_comb = split_comb
        idcs = split
        if phase!='test':
            idcs = [f for f in idcs if f not in self.blacklist]

        self.channel = config['chanel']
        if self.channel==2:
            self.filenames = [os.path.join(data_dir, '%s_merge.npy' % idx) for idx in idcs]
        elif self.channel ==1:
            if 'cleanimg' in config and  config['cleanimg']:
                self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
            else:
                self.filenames = [os.path.join(data_dir, '%s_img.npy' % idx) for idx in idcs]
        self.kagglenames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])>20]
        self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])<20]
        
        labels = []
        
        for idx in idcs:
            if config['luna_raw'] ==True:
                try:
                    l = np.load(os.path.join(data_dir, '%s_label_raw.npy' % idx))
                except:
                    try:
                        l = np.load(os.path.join(data_dir, '%s_label.npy' %idx))
                    except Exception as inst:
                        print (inst)
                        print("Skiping....%s" %idx)
                        continue
            else:
                try:
                    l = np.load(os.path.join(data_dir, '%s_label.npy' %idx))
                except Exception as inst:
                    print (inst)
                    print("Skiping....%s" % idx)
                    continue
            labels.append(l)

        self.sample_bboxes = labels
        if self.phase!='test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0 :
                    for t in l:
                        if t[3]>sizelim:
                            self.bboxes.append([np.concatenate([[i],t])])
                        if t[3]>sizelim2:
                            self.bboxes+=[[np.concatenate([[i],t])]]*2
                        if t[3]>sizelim3:
                            self.bboxes+=[[np.concatenate([[i],t])]]*4
            self.bboxes = np.concatenate(self.bboxes,axis = 0)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        isRandomImg  = False
        if self.phase !='test':
            if idx>=len(self.bboxes):
                isRandom = True
                idx = idx%len(self.bboxes)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False
        
        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)[0:self.channel]
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes,isScale,isRandom)
                if self.phase=='train' and not isRandom:
                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                        ifflip = self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)[0:self.channel]
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
            label = self.label_mapping(sample.shape[1:], target, bboxes)
            sample = sample.astype(np.float32)
            #if filename in self.kagglenames:
	        #	label[label==-1]=0
            sample = (sample.astype(np.float32)-128)/128
            return tf.stack(sample),tf.stack(label), coord
        else:
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)
            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
            imgs, nzhw = self.split_comb.split(imgs)
            coord2, nzhw2 = self.split_comb.split(coord,
                                                   side_len = self.split_comb.side_len/self.stride,
                                                   max_stride = self.split_comb.max_stride/self.stride,
                                                   margin = self.split_comb.margin/self.stride)
#            assert np.all(nzhw==nzhw2)
            imgs = (imgs.astype(np.float32)-128)/128
            return imgs,bboxes, coord2.astype(np.float32),\
                   np.array(nzhw)


    def __len__(self):
        if self.phase == 'train':
            return len(self.bboxes)/(1-self.r_rand)
        elif self.phase =='val':
            return len(self.bboxes)
        else:
            return len(self.filenames)