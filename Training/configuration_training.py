from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Define data dir configurations
__C.DIR = edict()
__C.DIR.stage1_data_path = '/Users/xuan/lung_cancer_data/sample_images'
__C.DIR.luna_raw = '/Users/xuan/lung_cancer_data/luna_raw'
__C.DIR.luna_segment = '/Users/xuan/lung_cancer_data/seg-lungs-LUNA16'
__C.DIR.luna_data = '/Users/xuan/lung_cancer_data/luna/allset'
__C.DIR.preprocess_result_path = '/Users/xuan/lung_cancer_data/preprocess_result'
__C.DIR.luna_abbr = '/Users/xuan/lung_cancer_data/luna_extra/shorter.csv'
__C.DIR.luna_label = '/Users/xuan/lung_cancer_data/luna_extra/lunaqualified.csv'
__C.DIR.stage1_annos_path = ['/Users/xuan/lung_cancer_data/stage_1_anno/labels/label_job5.csv',
                             '/Users/xuan/lung_cancer_data/stage_1_anno/labels/label_job4_2.csv',
                             '/Users/xuan/lung_cancer_data/stage_1_anno/labels/label_job4_1.csv',
                             '/Users/xuan/lung_cancer_data/stage_1_anno/labels/label_job0.csv',
                             '/Users/xuan/lung_cancer_data/stage_1_anno/labels/label_qualified.csv']
__C.DIR.bbox_path = '/Users/xuan/lung_cancer_data/results/res18/bbox/'
__C.DIR.save_dir = '/Users/xuan/lung_cancer_data/save'
__C.DIR.preprocessing_backend = 'python'

# HyperParameters for Training.
__C.TRAIN = edict()
__C.TRAIN.LEARNING_RATE = 0.001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = 50000
__C.TRAIN.DISPLAY = 10
__C.IS_MULTISCALE = False
__C.TRAIN.BATCH_SIZE = 128

