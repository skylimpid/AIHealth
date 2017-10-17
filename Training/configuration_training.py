from easydict import EasyDict as edict
from Training.constants import DATA_BASE_DIR, SYS_DIR

__C = edict()
cfg = __C

# Define data dir configurations
__C.DIR = edict()

# The directory which has the kaggle raw data
__C.DIR.kaggle_data_path = DATA_BASE_DIR + 'sample_images'

# The directory which has the luna raw data
__C.DIR.luna_raw = DATA_BASE_DIR + 'luna_raw'

# The directory which has the luna segment raw data
__C.DIR.luna_segment_raw = DATA_BASE_DIR + 'seg-lungs-LUNA16'

# The directory which will be used to save the preprocessed intermediate luna data
__C.DIR.luna_intermediate_data = DATA_BASE_DIR + 'luna_intermediate/luna_data'

# The directory which will be used to save the preprocessed intermediate luna segment data
__C.DIR.luna_intermediate_segment_data = DATA_BASE_DIR + 'luna_intermediate/luna_segment'

# The directory which will be used to save the preprocessed kaggle data and luna data
__C.DIR.preprocess_result_path = DATA_BASE_DIR + 'preprocess_result'

# Define several manual label files which will be used to do the preprocessing.
# All the csv file could be found under the directory ~/AIHealth/Training/Detector/labels
__C.DIR.luna_abbr = DATA_BASE_DIR + 'luna_extra/shorter.csv'
__C.DIR.luna_label = DATA_BASE_DIR + 'luna_extra/lunaqualified.csv'
__C.DIR.kaggle_annos_path = [DATA_BASE_DIR + 'kaggle_anno/labels/label_job5.csv',
                             DATA_BASE_DIR + 'kaggle_anno/labels/label_job4_2.csv',
                             DATA_BASE_DIR + 'kaggle_anno/labels/label_job4_1.csv',
                             DATA_BASE_DIR + 'kaggle_anno/labels/label_job0.csv',
                             DATA_BASE_DIR + 'kaggle_anno/labels/label_qualified.csv']
__C.DIR.kaggle_full_labels = DATA_BASE_DIR + 'kaggle_anno/labels/full_label.csv'

__C.DIR.bbox_path = DATA_BASE_DIR + 'results/bbox/'

__C.DIR.save_dir = DATA_BASE_DIR + 'save'

__C.DIR.preprocessing_backend = 'python'

__C.DIR.detector_net_saver_dir = SYS_DIR + "/Trained_Models/DetectorNetWeight"
__C.DIR.detector_net_saver_file_prefix = "detector_model_weights"

__C.DIR.detector_net_train_data_path = DATA_BASE_DIR + "full.npy"

# HyperParameters for Training.
__C.TRAIN = edict()
__C.TRAIN.LEARNING_RATE = 0.01
__C.TRAIN.LEARNING_RATE_STEP_SIZE = 100
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.DISPLAY_STEPS = 2
__C.TRAIN.BATCH_SIZE = 2
__C.TRAIN.EPOCHS = 10
__C.TRAIN.SAVE_STEPS = 100

