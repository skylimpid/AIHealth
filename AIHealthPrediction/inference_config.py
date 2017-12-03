from Training.constants import DIMEN_X, SYS_DIR
import os

config = {}
config['stride'] = 4
config['max_stride'] = 16
config['pad_value'] = 170
config['filling_value'] = 160
config['topk'] = 5
config['crop_size'] = [DIMEN_X, DIMEN_X, DIMEN_X]
config['conf_th'] = 0
config['nms_th'] = 0.05
config['detector_net_batch_size'] = 5
config['detector_net_ckg'] = os.path.join(SYS_DIR + "/Trained_Models/DetectorNetWeight")
config['classifier_net_ckg'] = os.path.join(SYS_DIR + "/Trained_Models/ClassifierNetWeight")

config['report_dir'] = '/home/xuan/AIHealthData/report'
