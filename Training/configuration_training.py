from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Define data dir configurations

DATA_BASE_DIR = "/home/xuan/lung_cancer_data/"
#DATA_BASE_DIR = "/Users/junmaa/workspace/Kaggle_NDSB/data/"

__C.DIR = edict()
__C.DIR.stage1_data_path = DATA_BASE_DIR + 'sample_images'
__C.DIR.luna_raw = DATA_BASE_DIR + 'luna_raw'
__C.DIR.luna_segment_raw = DATA_BASE_DIR + 'seg-lungs-LUNA16'
__C.DIR.luna_data = DATA_BASE_DIR + 'luna/allset/luna_data'
__C.DIR.luna_segment_data = DATA_BASE_DIR + 'luna/allset/luna_segment'
__C.DIR.preprocess_result_path = DATA_BASE_DIR + 'preprocess_result'
__C.DIR.luna_abbr = DATA_BASE_DIR + 'luna_extra/shorter.csv'
__C.DIR.luna_label = DATA_BASE_DIR + 'luna_extra/lunaqualified.csv'
__C.DIR.stage1_annos_path = [DATA_BASE_DIR + 'stage_1_anno/labels/label_job5.csv',
                             DATA_BASE_DIR + 'stage_1_anno/labels/label_job4_2.csv',
                             DATA_BASE_DIR + 'stage_1_anno/labels/label_job4_1.csv',
                             DATA_BASE_DIR + 'stage_1_anno/labels/label_job0.csv',
                             DATA_BASE_DIR + 'stage_1_anno/labels/label_qualified.csv']
__C.DIR.bbox_path = DATA_BASE_DIR + 'results/res18/bbox/'
__C.DIR.save_dir = DATA_BASE_DIR + 'save'
__C.DIR.preprocessing_backend = 'python'

# HyperParameters for Training.
__C.TRAIN = edict()
__C.TRAIN.LEARNING_RATE = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.DISPLAY = 10
__C.IS_MULTISCALE = False
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.EPOCHS = 100

