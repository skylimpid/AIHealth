import numpy as np
from Cython.Build import cythonize


def iou(box0, box1):
    cdef float r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    cdef float r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    cdef float intersection = overlap[0] * overlap[1] * overlap[2]
    cdef float union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    # print("intersection:{} and union:{}".format(intersection, union))
    return intersection / union


def nms(output, nms_th, valid_size=None):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    cdef int output_length = len(output)
    cdef int flag = 1

    for i in range(output_length):

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