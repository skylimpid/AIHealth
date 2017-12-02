from Training.constants import DIMEN_X, SYS_DIR

config = {}
config['stride'] = 4
config['max_stride'] = 16
config['pad_value'] = 170
config['topk'] = 5
config['crop_size'] = [DIMEN_X, DIMEN_X, DIMEN_X]
config['nms_th'] = 0.05
config['detector_net_batch_size'] = 5
config['detector_net_ckg'] = SYS_DIR + "/Trained_Models/DetectorNetWeight"
