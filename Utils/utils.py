import numpy as np


def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    # print("intersection:{} and union:{}".format(intersection, union))
    return intersection / union


class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output, thresh=-3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx, yy, zz, aa = np.where(mask)

        output = output[xx, yy, zz, aa]
        if ismask:
            return output, [xx, yy, zz, aa]
        else:
            return output

            # output = output[output[:, 0] >= self.conf_th]
            # bboxes = nms(output, self.nms_th)

def nms(output, nms_th, valid_size=None):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
            if valid_size is not None and len(bboxes) >= valid_size:
                break

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def acc(pbb, lbb, conf_th, nms_th, detect_th):
    pbb = pbb[pbb[:, 0] >= conf_th]
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score>bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p,[bestscore]],0))
            else:
                fp.append(np.concatenate([p,[bestscore]],0))
        if flag == 0:
            fp.append(np.concatenate([p,[bestscore]],0))
    for i,l in enumerate(lbb):
        if l_flag[i]==0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5],l))
            if len(score)!=0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore<detect_th:
                fn.append(np.concatenate([l,[bestscore]],0))

    return tp, fp, fn, len(lbb)