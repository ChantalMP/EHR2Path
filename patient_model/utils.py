import json
import os
import random
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm

from mimic_iv_extraction.paths import mimiciv_path

''' ICD category extraction code from: https://github.com/mmcdermott/comprehensive_MTL_EHR/blob/95bf1f6cea37299f6275df3a9066e19031173b3e/latent_patient_trajectories/representation_learner/extractors.py#L637'''
def get_icd_category(icd9_codes):
    cols = ['infection', 'neoplasms', 'endocrine', 'blood', 'mental', 'nervous', 'circulatory', 'respiratory', 'digestive',
            'genitourinary', 'pregnancy', 'skin', 'musculoskeletal', 'congenital', 'perinatal', 'ill_defined', 'injury', 'unknown']

    result = [0] * 18
    for code in icd9_codes:
        if 'v' in code.lower():
            continue
        if 'e' in code.lower():
            continue
        if len(code) > 3:
            code = code[:3] + '.' + code[4:]

        try:
            code = float(code)
        except:
            print(f"Error converting code to float: {code}")
            continue

        if (0 <= code) & (code < 140):
            # infection
            result[0] = 1
        if (140 <= code) & (code < 240):
            # neoplasms
            result[1] = 1
        if (240 <= code) & (code < 280):
            # endocrine/nutritional/metabolic
            result[2] = 1
        if (280 <= code) & (code < 290):
            # diseases of blood and blood forming organs
            result[3] = 1
        if (290 <= code) & (code < 319):
            # Mental disorders
            result[4] = 1
        if (320 <= code) & (code < 390):
            # Diseases of Nervous System and Sense Organs (PDF, 52KB)
            result[5] = 1
        if (390 <= code) & (code < 460):
            # Diseases of the Circulatory System (PDF, 23KB)
            result[6] = 1
        if (460 <= code) & (code < 520):
            # Diseases of the Respiratory System (PDF, 16KB)
            result[7] = 1
        if (520 <= code) & (code < 580):
            # Diseases of the Digestive System (PDF, 24KB)
            result[8] = 1
        if (580 <= code) & (code < 630):
            # Diseases of the Genitourinary System (PDF, 26KB)
            result[9] = 1
        if (630 <= code) & (code < 680):
            # Complications of Pregnancy, Childbirth and the Puerperium (PDF, 30KB)
            result[10] = 1
        if (680 <= code) & (code < 710):
            # Diseases of the Skin and Subcutaneous Tissue (PDF, 13KB)
            result[11] = 1
        if (710 <= code) & (code < 740):
            # Diseases of the Musculoskeletal System and Connective Tissue (PDF, 25KB)
            result[12] = 1
        if (740 <= code) & (code < 760):
            # Congenital Anomalies (PDF, 208KB)
            result[13] = 1
        if (760 <= code) & (code < 780):
            # Certain Conditions Originating in the Perinatal Period (PDF, 201KB)
            result[14] = 1
        if (780 <= code) & (code < 800):
            # Symptoms, Signs and Ill-defined Conditions (PDF, 209KB)
            result[15] = 1
        if (800 <= code) & (code < 1000):
            # Injury and Poisoning (PDF, 1.2MB)
            result[16] = 1
    if sum(result) == 0:
        result[17] = 1  # unknown or other col

    # convert to category names
    result = [cols[i] for i in range(18) if result[i] == 1]

    return result

def create_train_val_test_split():

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    # create dict from each elem in data to its patient_id saved in stay_to_ids_dict[stay_id][patient_id]
    patient_id_to_stay = defaultdict(list)
    for stay_id, patient_stay in stay_to_ids_dict.items():
        patient_id = patient_stay['patient_id']
        patient_id_to_stay[patient_id].append(stay_id)

    # split patient_ids into train, val, test on patient level
    patient_ids = list(patient_id_to_stay.keys())
    n = len(patient_ids)
    n_train = int(n * 0.95)
    n_val = int(n * 0.025) # half of the remaining data
    n_test = n - n_train - n_val

    # shuffle
    random.shuffle(patient_ids)
    train_patient_ids = patient_ids[:n_train]
    val_patient_ids = patient_ids[n_train:n_train+n_val]
    test_patient_ids = patient_ids[n_train+n_val:]

    # create train, val, test splits
    train_stay_ids = []
    val_stay_ids = []
    test_stay_ids = []

    for patient_id in train_patient_ids:
        train_stay_ids.extend(patient_id_to_stay[patient_id])
    for patient_id in val_patient_ids:
        val_stay_ids.extend(patient_id_to_stay[patient_id])
    for patient_id in test_patient_ids:
        test_stay_ids.extend(patient_id_to_stay[patient_id])

    train_data = set(["{}/stay_{}.json".format(mimiciv_path + "all_data_24_hours_unweighted", stay_id) for stay_id in train_stay_ids])
    val_data = set(["{}/stay_{}.json".format(mimiciv_path + "all_data_24_hours_unweighted", stay_id) for stay_id in val_stay_ids])
    test_data = set(["{}/stay_{}.json".format(mimiciv_path + "all_data_24_hours_unweighted", stay_id) for stay_id in test_stay_ids])

    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")

    # check which data is available in all_data folder
    data_paths = set(["{}/{}".format( mimiciv_path+ "all_data_24_hours_unweighted", f) for f in os.listdir(mimiciv_path + "all_data_24_hours_unweighted")])

    train_data = list(train_data.intersection(data_paths))
    val_data = list(val_data.intersection(data_paths))
    test_data = list(test_data.intersection(data_paths))

    with open(f"{mimiciv_path}train_paths.json", "w") as f:
        json.dump(train_data, f)
    with open(f"{mimiciv_path}val_paths.json", "w") as f:
        json.dump(val_data, f)
    with open(f"{mimiciv_path}test_paths.json", "w") as f:
        json.dump(test_data, f)


if __name__ == '__main__':
    create_train_val_test_split()