import numpy as np
import os
import shutil
from Utils.utils import nms, iou
from Training.configuration_training import cfg
from Net.tensorflow_model.classifier_net import get_config


def main(config, split, bboxpath):
    if os.path.exists(cfg.DIR.classifier_net_intermediate_candidate_box):
        shutil.rmtree(cfg.DIR.classifier_net_intermediate_candidate_box)
    else:
        os.makedirs(cfg.DIR.classifier_net_intermediate_candidate_box)

    if os.path.exists(cfg.DIR.classifier_net_intermediate_pbb_label):
        shutil.rmtree(cfg.DIR.classifier_net_intermediate_pbb_label)
    else:
        os.makedirs(cfg.DIR.classifier_net_intermediate_pbb_label)

    idcs = np.load(split)
    candidate_box = []
    pbb_label_total = []
    for idx in idcs.tolist():
        print(idx)
        pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
        pbb = pbb[pbb[:, 0] > config['conf_th']]
        pbb = nms(pbb, config['nms_th'], config['topk']*1000)
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
        candidate_box.append(pbb)
        pbb_label_total.append(np.array(pbb_label))
        candidate_box_file = os.path.join(cfg.DIR.classifier_net_intermediate_candidate_box, idx + "_candidate.npy")
        pbb_file = os.path.join(cfg.DIR.classifier_net_intermediate_pbb_label, idx + "_pbb.npy")
        np.save(candidate_box_file, candidate_box)
        np.save(pbb_file, pbb_label_total)


if __name__ == "__main__":
    config = get_config()
    main(config=config, split=cfg.DIR.classifier_net_train_data_path, bboxpath=cfg.DIR.bbox_path)