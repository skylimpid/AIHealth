import time
import numpy as np

class InferenceDataSet(object):
    def __init__(self, config, split_combiner, preprocessed_clean_img):
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
        self.topk = config['topk']
        self.crop_size = config['crop_size']
        self.split_comber = split_combiner
        self.clean_img = preprocessed_clean_img


    def getDetectorDataSet(self):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        imgs = self.clean_img
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
        return imgs, coord2, np.array(nzhw)

    def getClassiferDataSet(self, bbox):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        pbb = bbox
        conf_list = pbb[:, 0]
        topk = self.topk
        img = self.clean_img

        chosenid = conf_list.argsort()[::-1][:topk]
        croplist = np.zeros([topk, 1, self.crop_size[0], self.crop_size[1], self.crop_size[2]]).astype('float32')
        coordlist = np.zeros([topk, 3, self.crop_size[0] / self.stride, self.crop_size[1] / self.stride,
                              self.crop_size[2] / self.stride]).astype('float32')

        for i, id in enumerate(chosenid):
            target = pbb[id, 1:]
            crop, coord = self.crop(img, target)
            crop = crop.astype(np.float32)
            croplist[i] = crop
            coordlist[i] = coord


        return np.asarray(croplist), np.asarray(coordlist)
