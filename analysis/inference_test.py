import time
import pandas
import numpy as np

from AIHealthPrediction.inference import preprocess_and_inference
from Training.configuration_training import cfg
from Training.constants import DATA_BASE_DIR


def test_inference(labelfile, patient_id_file, dicom_dir, confidence_level):
    start = time.time()

    yset = {}
    labels = np.array(pandas.read_csv(labelfile))
    for i in range(len(labels)):
        yset[labels[i][0]] = labels[i][1]

    patient_ids = np.load(patient_id_file)
    total_num = len(patient_ids)
    print("Start to inference patients: ", total_num)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    count = 0
    for patient_id in patient_ids:
        predict, bbox, _, _, _ = preprocess_and_inference(patient_id,
                                                          dicom_dir,
                                                          confidence_level)

        if predict > 0.5:
            if yset[patient_id] > 0.5:
                true_pos += 1
                print("Result: true pos for ", patient_id)
            else:
                false_pos += 1
                print("Result: false pos for ", patient_id)
        else:
            if yset[patient_id] > 0.5:
                false_neg += 1
                print("Result: false neg for ", patient_id)
            else:
                true_neg += 1
                print("Result: true neg for ", patient_id)

        count += 1
        if count % 10 == 0:
            print("Step %d report: true pos: %d (%f), true neg: %d (%f), false pos: %d (%f), false neg: %d (%f)" % (
                                                                        count,
                                                                        true_pos, true_pos / count,
                                                                        true_neg, true_neg / count,
                                                                        false_pos, false_pos / count,
                                                                        false_neg, false_neg / count
                                                                    ))

    print("Total inference patient number: ", total_num)
    print("Stat report: true pos: %d (%f), true neg: %d (%f), false pos: %d (%f), false neg: %d (%f)" % (
                                                                            true_pos, true_pos/total_num,
                                                                            true_neg, true_neg/total_num,
                                                                            false_pos, false_pos/total_num,
                                                                            false_neg, false_neg/total_num
                                                                            ))

    end = time.time()
    print("The total time spend is {}".format(str(end - start)))

if __name__ == "__main__":
    kaggle_full_label_file = cfg.DIR.kaggle_full_labels
    patient_id_file = DATA_BASE_DIR + "/train_split/kaggle_validate_data.npy"
    dicom_dir = DATA_BASE_DIR + "/stage1"
    test_inference(kaggle_full_label_file, patient_id_file, dicom_dir, 0.5)