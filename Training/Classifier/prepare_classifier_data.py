import os
import shutil
import numpy as np
from Utils.utils import nms, iou
from Training.configuration_training import cfg
from Net.tensorflow_model.classifier_net import get_config


def main(config, split, bboxpath, clear = False):

    if clear:
        if os.path.exists(cfg.DIR.classifier_net_intermediate_candidate_box):
            shutil.rmtree(cfg.DIR.classifier_net_intermediate_candidate_box)
        if os.path.exists(cfg.DIR.classifier_net_intermediate_pbb_label):
            shutil.rmtree(cfg.DIR.classifier_net_intermediate_pbb_label)

    if not os.path.exists(cfg.DIR.classifier_net_intermediate_candidate_box):
        os.makedirs(cfg.DIR.classifier_net_intermediate_candidate_box)

    if not os.path.exists(cfg.DIR.classifier_net_intermediate_pbb_label):
        os.makedirs(cfg.DIR.classifier_net_intermediate_pbb_label)

    idcs = np.load(split)
    for idx in idcs.tolist():
        print(idx)
        pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
        pbb = pbb[pbb[:, 0] > config['conf_th']]
        #pbb = nms(pbb, config['nms_th'], config['topk']*1000)
        pbb = nms(pbb, config['nms_th'])
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
        pbb_label = np.array(pbb_label)
        candidate_box_file = os.path.join(cfg.DIR.classifier_net_intermediate_candidate_box, idx + "_candidate.npy")
        pbb_file = os.path.join(cfg.DIR.classifier_net_intermediate_pbb_label, idx + "_pbb.npy")
        np.save(candidate_box_file, pbb)
        np.save(pbb_file, pbb_label)

    print("Prepared totally: ", len(idcs))


if __name__ == "__main__":
    config = get_config()
    # please use the following path appropriately for preparing:
    # training data path: cfg.DIR.classifier_net_train_data_path
    # validation data path: cfg.DIR.detector_net_validate_data_path
    main(config=config, split=cfg.DIR.detector_net_validate_data_path, bboxpath=cfg.DIR.bbox_path)