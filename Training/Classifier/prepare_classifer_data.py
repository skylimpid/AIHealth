import numpy as np
import os
from Utils.utils import nms, iou
from Training.configuration_training import cfg
from Net.tensorflow_model.ClassiferNet import get_config


def main(config, split, bboxpath):
    if os.path.exists(cfg.DIR.classifier_net_intermediate_candidate_box):
        os.remove(cfg.DIR.classifier_net_intermediate_candidate_box)
    else:
        os.makedirs(cfg.DIR.classifier_net_intermediate_candidate_box, exist_ok=True)

    if os.path.exists(cfg.DIR.classifier_net_intermediate_pbb_label):
        os.remove(cfg.DIR.classifier_net_intermediate_pbb_label)
    else:
        os.makedirs(cfg.DIR.classifier_net_intermediate_pbb_label, exist_ok=True)

    idcs = np.load(split)
    idcs = [f.split('-')[0] for f in idcs]
    candidate_box = []
    pbb_label_total = []
    for idx in idcs:
        print(idx)
        pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
        pbb = pbb[pbb[:, 0] > config['conf_th']]
        pbb = nms(pbb, config['nms_th'], config['topk'] * 100)
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

    np.save(cfg.DIR.classifier_net_intermediate_candidate_box, candidate_box)
    np.save(cfg.DIR.classifier_net_intermediate_pbb_label, pbb_label_total)




if __name__ == "__main__":
    config = get_config()
    main(config=config, split=cfg.DIR.classifier_net_train_data_path, bboxpath=cfg.DIR.bbox_path)