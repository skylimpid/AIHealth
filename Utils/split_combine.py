import numpy as np


class SplitComb():
    def __init__(self, side_len, max_stride, stride, margin, pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
        
    def split(self, data, side_len = None, max_stride = None, margin = None):

        if side_len is None:
            side_len = self.side_len

        if max_stride is None:
            max_stride = self.max_stride

        if margin is None:
            margin = self.margin

        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))

        self.nzhw = [nz, nh, nw]
        
        pad = [[0, 0],
               [margin, nz * side_len - z + margin],
               [margin, nh * side_len - h + margin],
               [margin, nw * side_len - w + margin]]

        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)

        return splits, self.nzhw

    def combine(self, output, nzhw=None, side_len=None, stride=None, margin=None):
        
        if side_len is None:
            side_len = self.side_len

        if stride is None:
            stride = self.stride

        if margin is None:
            margin = self.margin

        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw

        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len = int(side_len/stride)
        margin = int(margin/stride)

        splits = []
        for i in range(output.get_shape().as_list()[0]):
            splits.append(output[i])

        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len
                   # print("idx:" + str(idx))
                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew, :, :] = split.eval()
                    idx += 1

        return output 
