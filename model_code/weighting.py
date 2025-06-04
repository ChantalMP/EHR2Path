import argparse
import gzip
import json
import random
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from mimic_iv_extraction.paths import mimiciv_path
from patient_model.retrieve_patient_model import retrieve_patient_model_, restrict_patient_until_hour
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_paths(d, parent_key=None):
    paths = []
    parent_key = parent_key or []

    for k, v in d.items():
        current_path = parent_key + [k]

        if isinstance(v, dict):
            paths.extend(extract_paths(v, current_path))
        else:
            # check if outcome task
            new_path = '.'.join(current_path)
            if k == 'Disposition' and 'MDProgressNote' not in new_path:
                paths.append(new_path + '.' + v)
            elif k == 'ICD categories':
                for icd in v.split(';'):
                    paths.append(new_path + '.' + icd)
            else:
                paths.append(new_path)

    return paths


def generate_key_counts(patient_path):
    patient_path = f"{mimiciv_path}all_data_24_hours_unweighted/{patient_path.split('/')[-1]}"
    sample = json.load(open(patient_path, "r"))

    path_counts = Counter()
    for timepoint in sample:
        desc = timepoint['desc']
        change_log = timepoint['change_log']
        # extract all occuring keys
        paths = extract_paths(change_log)
        # include timepoints, after ICU discharge and after Hospital admission, so that the model can learn to predict what follows these events
        if 'Hospital Stay' in desc and 'Disposition' in desc['Hospital Stay'] and desc['Hospital Stay']['Disposition'] == 'DISCHARGED FROM ICU':
            paths.append('Hospital Stay.Disposition.DISCHARGED FROM ICU')
        if 'Emergency Department Stay' in desc and 'Disposition' in desc['Emergency Department Stay'] and desc['Emergency Department Stay']['Disposition'] == 'ADMITTED':
            paths.append('Emergency Department Stay.Disposition.JUST ADMITTED')
        path_counts.update(paths)

    return path_counts


def extract_event_counts():
    patient_paths = json.load(open(f"{mimiciv_path}train_paths.json", "r"))

    with Pool(1) as pool:
        counters = list(tqdm(pool.imap(generate_key_counts, patient_paths), total=len(patient_paths)))

    # merge all counts
    all_event_counts = Counter()
    for counter in counters:
        all_event_counts.update(counter)

    # save the counts
    with open(f"{mimiciv_path}train_event_counts.json", "w") as f:
        json.dump(all_event_counts, f)


def calculate_patient_weights(patient_path, rel_freq, max_freq, frequency_alpha=3., epsilon=1e-10):
    sample_weights = []
    # sample_types = []
    sample_paths = []
    patient_path = f"{mimiciv_path}all_data_24_hours_unweighted/{patient_path.split('/')[-1]}"
    patient_sample = json.load(open(patient_path, "r"))
    patient_path = patient_path.split('/')[-1]
    for timepoint, sample in enumerate(patient_sample):
        paths = extract_paths(sample['change_log'])
        weights = []
        icd_weights = []
        for event in paths:
            # check if outcome task
            freq = rel_freq.get(event, max_freq) + epsilon
            if 'Disposition' in event and 'MDProgressNote' not in event:
                if 'Hospital Stay' in event:
                    highest_freq = rel_freq.get('Hospital Stay.Disposition.DISCHARGED', max_freq) + epsilon
                    regular_highest_weight = (-np.log(highest_freq))
                elif 'Emergency Department Stay' in event:
                    highest_freq = rel_freq.get('Emergency Department Stay.Disposition.DISCHARGED', max_freq) + epsilon
                    regular_highest_weight = (-np.log(highest_freq))
                elif 'ICU Stay' in event:
                    # only one possible disposition event, no explicit value weighting needed
                    weight = (-np.log(freq))
                    icd_weights.append(weight)
                    continue

                event_freq = rel_freq.get(event, max_freq) + epsilon
                # select event weight so that is will be sampled the same as regular discharge (linear weighting)
                weight = regular_highest_weight * (highest_freq / event_freq) #linear weighting, but slowed down by ICD code interactions
                # weight = (-np.log(event_freq))
                weights.append(weight)

            elif 'ICD categories' in event:
                # weight of Hospital Stay.ICD categories.circulatory:
                if 'Hospital Stay' in event:
                    highest_freq = rel_freq.get('Hospital Stay.ICD categories.circulatory', max_freq) + epsilon
                    regular_highest_weight = (-np.log(highest_freq))
                elif 'Emergency Department Stay' in event:
                    highest_freq = rel_freq.get('Emergency Department Stay.ICD categories.unknown', max_freq) + epsilon
                    regular_highest_weight = (-np.log(highest_freq))
                else:
                    raise ValueError("ICD category not in Hospital Stay or Emergency Department Stay")
                event_freq = rel_freq.get(event, max_freq) + epsilon
                # select event weight so that is will be sampled the same as regular discharge
                weight = (regular_highest_weight * (highest_freq / event_freq))#**0.5 #square root weighting
                # weight = (-np.log(event_freq)) #logarithmic weighting
                icd_weights.append(weight/2) # reduce influence of ICD codes in comparison to Disposition events

            else:
                # Use logarithmic weighting to prevent extreme weights
                weight = -np.log(freq)  # ==1: logaritmic weighting, >1: more extreme weighting, <1: less extreme weighting
                weights.append(weight)

        # calculate sample weight as mean of all event weights
        if len(icd_weights) > 0: # first calculate one weight expressing the "interestingness" of the diagnosis codes
            icd_weight = (np.mean(np.array(icd_weights) ** frequency_alpha)) ** (1 / frequency_alpha)
            weights.append(icd_weight)

        # then merge with the weights of the Disposition event - this already is always only ONE event

        if len(weights) > 0:
            sample_weight = (np.mean(np.array(weights) ** frequency_alpha)) ** (1 / frequency_alpha)  # mean with emphasis on high values (=rare events) -> samples with many rare events are more important
        else:  # no changes in this timepoint
            sample_weight = -np.log(max_freq)

        # if 'Hospital Stay.Disposition.DISCHARGED' in paths:
        #     sample_types.append('Hospital Stay.Disposition.DISCHARGED')
        # elif 'Hospital Stay.Disposition.DIED' in paths:
        #     sample_types.append('Hospital Stay.Disposition.DIED')
        # else:
        #     sample_types.append('irrelevant')
        sample_weights.append(sample_weight)
        sample_paths.append(f"{patient_path}_{timepoint}")
    return sample_weights, sample_paths#, sample_types


def process_patient_path(args):
    patient_path, rel_freq, max_freq, frequency_alpha, epsilon = args
    return calculate_patient_weights(patient_path, rel_freq, max_freq, frequency_alpha, epsilon)


def calculate_weights(frequency_alpha=3., split='train'):
    patient_paths = json.load(open(f"{mimiciv_path}{split}_paths.json", "r"))
    event_counts = json.load(open(f"{mimiciv_path}train_event_counts.json", "r"))
    epsilon = 1e-10
    # calculate weights
    total_events = sum(event_counts.values())

    rel_freq = {k: v / total_events for k, v in event_counts.items()}
    max_freq = max(rel_freq.values()) + epsilon

    # Now, for each sample, compute a weight based on the frequencies of its relations
    sample_weights = []
    sample_paths = []
    # sample_types = []
    args = [(patient_path, rel_freq, max_freq, frequency_alpha, epsilon) for patient_path in patient_paths]

    with Pool(8) as pool:
        results = list(tqdm(pool.imap(process_patient_path, args), total=len(args)))

    for sample_weight, sample_path in results:
        sample_weights.extend(sample_weight)
        sample_paths.extend(sample_path)
        # sample_types.extend(sample_type)

    # Normalize the sample weights to sum to 1
    sample_weights = np.array(sample_weights)
    sample_weights /= sample_weights.sum()
    assert len(sample_weights) == len(sample_paths)

    # Save sample_paths with gzip compression
    with gzip.open(f"{mimiciv_path}{split}_sample_timepoint_paths.json.gz", "wt") as f:
        json.dump(sample_paths, f)

    # Save sample_weights with gzip compression
    with gzip.open(f"{mimiciv_path}{split}_sample_timepoint_weights.json.gz", "wt") as f:
        json.dump(sample_weights.tolist(), f)

    # save sample_types with gzip compression
    # with gzip.open(f"{mimiciv_path}{split}_sample_timepoint_types_DEBUG_linear.json.gz", "wt") as f:
    #     json.dump(sample_types, f)

def create_resampled_jsons(split='train', weighting=True, dataset_size=None):
    # load saved sample_paths and sample_weights
    sample_paths = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_paths.json.gz", "rt"))
    sample_weights = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_weights.json.gz", "rt"))
    # sample_types = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_types_DEBUG_linear.json.gz", "rt"))

    # Resample the samples using these weights
    # dataset_size = len(sample_paths)//20
    if dataset_size is None:
        dataset_size = len(sample_paths) // 20  # 20 is arbitrarily chosen
    if weighting:
        print("Weighting")
        resampled_samples = list(np.random.choice(sample_paths, size=dataset_size, replace=False, p=sample_weights))
        # resampled_types = [sample_types[sample_paths.index(path)] for path in resampled_samples]
        # Save the resampled samples with gzip compression
        with gzip.open(f"{mimiciv_path}{split}_weighting_sample_timepoint_paths_1Mio.json.gz", "wt") as f:
            json.dump(resampled_samples, f)

    else:
        print("No weighting")
        resampled_samples = list(np.random.choice(sample_paths, size=dataset_size, replace=False))
        # Save the resampled samples with gzip compression
        with gzip.open(f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz", "wt") as f:
            json.dump(resampled_samples, f)

    return resampled_samples

def init_worker(shared_dict):
    global stay_to_ids_dict
    stay_to_ids_dict = shared_dict

def process_sample_path_global(path_and_weight):
    global stay_to_ids_dict
    path, weight = path_and_weight

    sample_path, timepoint_idx = path.rsplit("_", 1)
    stay_id = sample_path[5:].split('.json')[0]
    complete_patient_stay = stay_to_ids_dict.get(stay_id)

    if (complete_patient_stay is None or
        'ed_stay_hours' not in complete_patient_stay or
        complete_patient_stay['ed_stay_hours'] is None or
        complete_patient_stay['ed_stay_hours'] == 0):
        return None # patient never in ED

    if int(timepoint_idx)-1 > complete_patient_stay['ed_stay_hours']:
        return None # patient currently not in ED

    try:
        full_patient_visit, _ = retrieve_patient_model_(
            complete_stay_id=stay_id,
            complete_patient_stay=complete_patient_stay,
            hour_idx=None
        )
    except Exception:
        return None

    final_disp = full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0]
    if final_disp not in ["ADMITTED", "HOME"]:
        return None

    return (path, 1 if final_disp == "ADMITTED" else 0, weight)

all_icd_codes = ['infection', 'neoplasms', 'endocrine', 'blood', 'mental', 'nervous', 'circulatory', 'respiratory', 'digestive',
                 'genitourinary', 'pregnancy', 'skin', 'musculoskeletal', 'congenital', 'perinatal', 'ill_defined', 'injury', 'unknown']

def process_sample_path_global_icd(path_and_weight):
    global stay_to_ids_dict
    path, weight = path_and_weight

    sample_path, timepoint_idx = path.rsplit("_", 1)

    stay_id = sample_path[5:].split('.json')[0]
    complete_patient_stay = stay_to_ids_dict.get(stay_id)

    if (complete_patient_stay is None or
        'ed_stay_hours' not in complete_patient_stay or
        complete_patient_stay['ed_stay_hours'] is None or
        complete_patient_stay['ed_stay_hours'] == 0):
        return None # patient never in ED

    if int(timepoint_idx)-1 != complete_patient_stay['ed_stay_hours']:
        return None # only need last timepoints in ED stays

    try:
        full_patient_visit, _ = retrieve_patient_model_(
            complete_stay_id=stay_id,
            complete_patient_stay=complete_patient_stay,
            hour_idx=None
        )
    except Exception:
        return None

    final_icd = full_patient_visit.patient_ed.icd_categories
    # covert to one-hot encoding
    one_hot_icd = [1 if code in final_icd else 0 for code in all_icd_codes]

    return (path, one_hot_icd, weight)

def process_sample_path_global_hosp_icd(path_and_weight):
    global stay_to_ids_dict
    # path, weight = path_and_weight
    path = path_and_weight

    sample_path, timepoint_idx = path.rsplit("_", 1)

    stay_id = sample_path[5:].split('.json')[0]
    complete_patient_stay = stay_to_ids_dict.get(stay_id)
    current_hour_idx = int(timepoint_idx) - 1

    if 'hosp_stay_hours' not in complete_patient_stay or complete_patient_stay['hosp_stay_hours'] == None or complete_patient_stay[
        'hosp_stay_hours'] <= 0:  # skip samples without Hosp stay
        return None

    if ('ed_stay_hours' in complete_patient_stay and
        complete_patient_stay['ed_stay_hours'] is not None and
        complete_patient_stay['ed_stay_hours'] > 0 and
        current_hour_idx <= complete_patient_stay['ed_stay_hours']):
        return None # patient currently in ED - skip

    # extract all admission timepoints
    target_timepoint_idx = complete_patient_stay['hosp_stay_hours'] + 1
    # if ed_stay need to add ed_stay_length to hour_idx
    if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] > 0:
        target_timepoint_idx += complete_patient_stay['ed_stay_hours'] + 2  # +2 for transition idxs

    # generate samples for all timepoints
    target_hour_idx = int(target_timepoint_idx) - 1

    if current_hour_idx != target_hour_idx:
        return None


    try:
        full_patient_visit, _ = retrieve_patient_model_(
            complete_stay_id=stay_id,
            complete_patient_stay=complete_patient_stay,
            hour_idx=None
        )
    except Exception:
        print("Error retrieving patient model")
        return None


    final_icd = full_patient_visit.patient_adm.icd_categories
    # covert to one-hot encoding
    one_hot_icd = [1 if code in final_icd else 0 for code in all_icd_codes]

    return (path, one_hot_icd)

def process_sample_path_global_mort(path):
    global stay_to_ids_dict

    sample_path, timepoint_idx = path.rsplit("_", 1)

    stay_id = sample_path[5:].split('.json')[0]

    complete_patient_stay = stay_to_ids_dict.get(stay_id)
    current_hour_idx = int(timepoint_idx) - 1

    if complete_patient_stay is None or 'icu_stay_ids' not in complete_patient_stay or len(complete_patient_stay['icu_stay_ids']) == 0:
        return None # patient never in ICU

    if ('ed_stay_hours' in complete_patient_stay and
        complete_patient_stay['ed_stay_hours'] is not None and
        complete_patient_stay['ed_stay_hours'] > 0 and
        current_hour_idx <= complete_patient_stay['ed_stay_hours']):
        return None # patient currently in ED - skip

    try:
        full_patient_visit, _ = retrieve_patient_model_(
            complete_stay_id=stay_id,
            complete_patient_stay=complete_patient_stay,
            hour_idx=None
        )

        _, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay=False, restrict_to_N_hours=24)

    except Exception:
        return None


    # check if the timepoint is in a ICU stay
    icu_outtime = None
    in_icu = False
    for idx, stay in enumerate(full_patient_visit.patient_icu):
        if stay.icustays.iloc[0]['intime'] < hour_time <= stay.icustays.iloc[0]['outtime']:  # current stay -> save outtime
            icu_outtime = stay.icustays.iloc[0]['outtime']
            in_icu = True
            break

    if not in_icu:
        return None

    final_hosp_disposition = "DIED" if full_patient_visit.patient_adm.admissions.discharge_location.values[0] == "DIED" else "DISCHARGED"
    if final_hosp_disposition == "DIED":
        # if died need to check if died within current ICU stay or not
        time_of_death = full_patient_visit.patient_adm.admissions.deathtime.values[0]
        if np.isnan(time_of_death):
            return None
        if icu_outtime is not None and time_of_death <= icu_outtime and time_of_death <= hour_time + pd.Timedelta(hours=24): # death within next 24h and before ICU discharge
            mort_label = 1
        else:
            mort_label = 0
    else:
        mort_label = 0

    return (path, mort_label)

def process_sample_path_global_los_path(path):
    global stay_to_ids_dict

    sample_path, timepoint_idx = path.rsplit("_", 1)

    stay_id = sample_path[5:].split('.json')[0]

    complete_patient_stay = stay_to_ids_dict.get(stay_id)
    current_hour_idx = int(timepoint_idx) - 1

    if complete_patient_stay is None or 'icu_stay_ids' not in complete_patient_stay or len(complete_patient_stay['icu_stay_ids']) == 0:
        return None # patient never in ICU

    if ('ed_stay_hours' in complete_patient_stay and
        complete_patient_stay['ed_stay_hours'] is not None and
        complete_patient_stay['ed_stay_hours'] > 0 and
        current_hour_idx <= complete_patient_stay['ed_stay_hours']):
        return None # patient currently in ED - skip

    try:
        full_patient_visit, _ = retrieve_patient_model_(
            complete_stay_id=stay_id,
            complete_patient_stay=complete_patient_stay,
            hour_idx=None
        )

        _, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay=False, restrict_to_N_hours=24)

    except Exception:
        return None

    # check if the timepoint is in a ICU stay and between hour 24 and 96
    icu_intime = None
    icu_outtime = None
    in_icu = False
    for idx, stay in enumerate(full_patient_visit.patient_icu):
        if stay.icustays.iloc[0]['intime'] < hour_time <= stay.icustays.iloc[0]['outtime']:  # current stay -> save outtime
            icu_intime = stay.icustays.iloc[0]['intime']
            icu_outtime = stay.icustays.iloc[0]['outtime']
            in_icu = True
            break

    if not in_icu:
        return None

    target_start_time = icu_intime + pd.Timedelta(hours=24)
    target_end_time = icu_intime + pd.Timedelta(hours=24) + pd.Timedelta(days=3)
    if not (target_start_time <= hour_time <= target_end_time):
        return None

    time_to_release = icu_outtime - hour_time
    # LOS > 3 days
    if time_to_release > pd.Timedelta(days=3):
        los_3day_label = 1
    else:
        los_3day_label = 0

    return (path, los_3day_label)

def process_sample_path_global_los_direct(path_and_weight):
    global stay_to_ids_dict
    path, weight = path_and_weight

    sample_path, timepoint_idx = path.rsplit("_", 1)

    stay_id = sample_path[5:].split('.json')[0]

    complete_patient_stay = stay_to_ids_dict.get(stay_id)
    current_hour_idx = int(timepoint_idx) - 1

    if complete_patient_stay is None or 'icu_stay_ids' not in complete_patient_stay or len(complete_patient_stay['icu_stay_ids']) == 0:
        return None # patient never in ICU

    if ('ed_stay_hours' in complete_patient_stay and
        complete_patient_stay['ed_stay_hours'] is not None and
        complete_patient_stay['ed_stay_hours'] > 0 and
        current_hour_idx <= complete_patient_stay['ed_stay_hours']):
        return None # patient currently in ED - skip

    try:
        full_patient_visit, _ = retrieve_patient_model_(
            complete_stay_id=stay_id,
            complete_patient_stay=complete_patient_stay,
            hour_idx=None
        )

        _, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay=False, restrict_to_N_hours=24)

    except Exception:
        return None

    # check if the timepoint is in a ICU stay and between hour 24 and 96
    icu_intime = None
    icu_outtime = None
    in_icu = False
    for idx, stay in enumerate(full_patient_visit.patient_icu):
        if stay.icustays.iloc[0]['intime'] < hour_time <= stay.icustays.iloc[0]['outtime']:  # current stay -> save outtime
            icu_intime = stay.icustays.iloc[0]['intime']
            icu_outtime = stay.icustays.iloc[0]['outtime']
            in_icu = True
            break

    if not in_icu:
        return None

    target_time = icu_intime + pd.Timedelta(hours=24)
    if hour_time != target_time:
        return None

    time_to_release = icu_outtime - hour_time
    # LOS > 3 days
    if time_to_release > pd.Timedelta(days=3):
        los_3day_label = 1
    else:
        los_3day_label = 0

    return (path, los_3day_label, weight)

def create_resampled_jsons_ed_adm_only(split='train', weighting=True, dataset_size=None):
    global stay_to_ids_dict
    # load saved sample_paths and sample_weights
    sample_paths = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_paths.json.gz", "rt"))
    sample_weights = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_weights.json.gz", "rt"))
    # sample_types = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_types_DEBUG_linear.json.gz", "rt"))

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    zipped_input = list(zip(sample_paths, sample_weights))

    n_workers = 200
    chunksize = 1000

    paths_to_keep = []
    labels = []
    weights = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(stay_to_ids_dict,)
    ) as executor:
        results = executor.map(process_sample_path_global, zipped_input, chunksize=chunksize)
        for result in tqdm(results, total=len(sample_paths)):
            if result:
                path, label, weight = result
                paths_to_keep.append(path)
                labels.append(label)
                weights.append(weight)

    print(f"Loaded {len(paths_to_keep)} paths, {len(labels)} labels, and {len(weights)} weights.")

    # normalize new weights to sum to 1
    weights = np.array(weights)
    weights /= weights.sum()
    assert len(weights) == len(paths_to_keep)

    if weighting:
        print("Weighting")
        dataset_size = min(dataset_size, len(paths_to_keep))
        resampled_samples = list(np.random.choice(paths_to_keep, size=dataset_size, replace=False, p=weights))
        # resampled_types = [sample_types[sample_paths.index(path)] for path in resampled_samples]
        # sample these such that labels are balanced

        path_to_label = dict(zip(paths_to_keep, labels))
        ones = [path for path in resampled_samples if path_to_label[path] == 1]
        zeros = [path for path in resampled_samples if path_to_label[path] == 0]
        minority_size = min(len(ones), len(zeros))
        balanced_samples = np.random.choice(ones, size=minority_size, replace=False).tolist() + \
                           np.random.choice(zeros, size=minority_size, replace=False).tolist()

        np.random.shuffle(balanced_samples)


        # Save the resampled samples with gzip compression
        with gzip.open(f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_ed_admission_all.json.gz", "wt") as f:
            json.dump(balanced_samples, f)

    else:
        print("No weighting")
        resampled_samples = list(np.random.choice(paths_to_keep, size=dataset_size, replace=False))

        path_to_label = dict(zip(paths_to_keep, labels))
        ones = [path for path in resampled_samples if path_to_label[path] == 1]
        zeros = [path for path in resampled_samples if path_to_label[path] == 0]
        minority_size = min(len(ones), len(zeros))
        balanced_samples = np.random.choice(ones, size=minority_size, replace=False).tolist() + \
                           np.random.choice(zeros, size=minority_size, replace=False).tolist()

        np.random.shuffle(balanced_samples)

        with gzip.open(f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_admission_all.json.gz", "wt") as f:
            json.dump(balanced_samples, f)

    return resampled_samples

def create_resampled_jsons_ed_icd_only(split='train', weighting=False, dataset_size=None):
    global stay_to_ids_dict
    # load saved sample_paths and sample_weights
    sample_paths = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_paths.json.gz", "rt")) # all timepoints of all paths -> get last ed timepoint samples for diagnosis
    sample_weights = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_weights.json.gz", "rt"))

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    zipped_input = list(zip(sample_paths, sample_weights))

    n_workers = 200
    chunksize = 1000

    paths_to_keep = []
    labels = []
    weights = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(stay_to_ids_dict,)
    ) as executor:
        results = executor.map(process_sample_path_global_icd, zipped_input, chunksize=chunksize)
        for result in tqdm(results, total=len(sample_paths)):
            if result:
                path, label, weight = result
                paths_to_keep.append(path)
                labels.append(label)
                weights.append(weight)


    print(f"Loaded {len(paths_to_keep)} paths, {len(labels)} labels, and {len(weights)} weights.")

    all_labels = np.stack(labels)  # shape: (N, 18)
    num_pos = np.sum(all_labels, axis=0)         # shape: (18,)
    num_neg = all_labels.shape[0] - num_pos
    pos_weight = num_neg / (num_pos + 1e-5)
    pos_weight = np.log1p(pos_weight)  # log(1 + num_neg / num_pos)

    # Compute sample-wise weights as the *mean* of active label weights
    weights = []
    for labels in all_labels:
        active_weights = pos_weight[labels == 1]
        if len(active_weights) == 0:
            weight = np.min(pos_weight)
        else:
            weight = np.mean(active_weights)
        weights.append(weight)

    # resample to 1
    weights /= np.array(weights).sum()
    print("Length resampling weights:", len(weights))

    resampled_samples = list(np.random.choice(paths_to_keep, size=dataset_size, replace=True, p=weights))

    with gzip.open(f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_icd_all.json.gz", "wt") as f:
        json.dump(resampled_samples, f)

    return resampled_samples

def create_resampled_jsons_icu_mort(split='train'):
    global stay_to_ids_dict
    # load normal weighted timepoints -> need to sort them out only keeping ICU timepoints and balancing by mortality label
    if split == 'train':
        with gzip.open(f"{mimiciv_path}train_weighting_sample_timepoint_paths_1Mio.json.gz","rt") as f:  # training data for 1 Mio sample WITH WEIGHTING
            sample_paths = json.load(f)
    else:
        with gzip.open(f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz","rt") as f:  # standard val data (val_data_len//20 -> 8300 samples)
            sample_paths = json.load(f)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    n_workers = 200
    chunksize = 1000

    paths_to_keep = []
    labels = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(stay_to_ids_dict,)
    ) as executor:
        results = executor.map(process_sample_path_global_mort, sample_paths, chunksize=chunksize)
        for result in tqdm(results, total=len(sample_paths)):
            if result:
                path, label = result
                paths_to_keep.append(path)
                labels.append(label)

    print(f"Loaded {len(paths_to_keep)} paths, {len(labels)} labels.")
    # balance by mortality label
    ones = [path for path in paths_to_keep if labels[paths_to_keep.index(path)] == 1]
    print("Number of positive samples:", len(ones))
    zeros = [path for path in paths_to_keep if labels[paths_to_keep.index(path)] == 0]
    minority_size = min(len(ones), len(zeros))
    balanced_samples = np.random.choice(ones, size=minority_size, replace=False).tolist() + \
                          np.random.choice(zeros, size=minority_size, replace=False).tolist()
    np.random.shuffle(balanced_samples)
    # Save the resampled samples with gzip compression
    with gzip.open(f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_icu_mort.json.gz", "wt") as f:
        json.dump(balanced_samples, f)

def create_resampled_jsons_icu_los_path(split='train'):
    global stay_to_ids_dict
    # load normal weighted timepoints -> need to sort them out only keeping ICU timepoints and balancing by mortality label
    if split == 'train':
        with gzip.open(f"{mimiciv_path}train_weighting_sample_timepoint_paths_1Mio.json.gz","rt") as f:  # training data for 1 Mio sample WITH WEIGHTING
            sample_paths = json.load(f)
    else:
        with gzip.open(f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz","rt") as f:  # standard val data (val_data_len//20 -> 8300 samples)
            sample_paths = json.load(f)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    n_workers = 200
    chunksize = 1000

    paths_to_keep = []
    labels = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(stay_to_ids_dict,)
    ) as executor:
        results = executor.map(process_sample_path_global_los_path, sample_paths, chunksize=chunksize)
        for result in tqdm(results, total=len(sample_paths)):
            if result:
                path, label = result
                paths_to_keep.append(path)
                labels.append(label)

    print(f"Loaded {len(paths_to_keep)} paths, {len(labels)} labels.")
    # balance by mortality label
    ones = [path for path in paths_to_keep if labels[paths_to_keep.index(path)] == 1]
    print("Number of positive samples:", len(ones))
    zeros = [path for path in paths_to_keep if labels[paths_to_keep.index(path)] == 0]
    minority_size = min(len(ones), len(zeros))
    balanced_samples = np.random.choice(ones, size=minority_size, replace=False).tolist() + \
                          np.random.choice(zeros, size=minority_size, replace=False).tolist()
    np.random.shuffle(balanced_samples)
    # Save the resampled samples with gzip compression
    with gzip.open(f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_icu_los_path.json.gz", "wt") as f:
        json.dump(balanced_samples, f)

def create_resampled_jsons_hosp_icd(split='train', weighting=False, dataset_size=None):
    global stay_to_ids_dict
    # load saved sample_paths and sample_weights
    # sample_paths = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_paths.json.gz", "rt")) # all timepoints of all paths -> get last ed timepoint samples for diagnosis
    # sample_weights = json.load(gzip.open(f"{mimiciv_path}{split}_sample_timepoint_weights.json.gz", "rt"))

    if split == 'train':
        with gzip.open(f"{mimiciv_path}train_weighting_sample_timepoint_paths_1Mio.json.gz","rt") as f:  # training data for 1 Mio sample WITH WEIGHTING
            sample_paths = json.load(f)
    else:
        with gzip.open(f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz","rt") as f:  # standard val data (val_data_len//20 -> 8300 samples)
            sample_paths = json.load(f)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    # zipped_input = list(zip(sample_paths, sample_weights))

    n_workers = 200
    chunksize = 1000

    paths_to_keep = []
    labels = []
    weights = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(stay_to_ids_dict,)
    ) as executor:
        results = executor.map(process_sample_path_global_hosp_icd, sample_paths, chunksize=chunksize) #zipped_input
        for result in tqdm(results, total=len(sample_paths)):
            if result:
                path, label = result
                paths_to_keep.append(path)
                labels.append(label)

    print(f"Loaded {len(paths_to_keep)} paths, {len(labels)} labels, and {len(weights)} weights.")

    if weighting:

        all_labels = np.stack(labels)  # shape: (N, 18)
        num_pos = np.sum(all_labels, axis=0)         # shape: (18,)
        num_neg = all_labels.shape[0] - num_pos
        pos_weight = num_neg / (num_pos + 1e-5)
        pos_weight = np.log1p(pos_weight)  # log(1 + num_neg / num_pos)

        # Compute sample-wise weights as the *mean* of active label weights
        weights = []
        for labels in all_labels:
            active_weights = pos_weight[labels == 1]
            if len(active_weights) == 0:
                weight = np.min(pos_weight)
            else:
                weight = np.mean(active_weights)
            weights.append(weight)

        # resample to 1
        weights /= np.array(weights).sum()
        print("Length resampling weights:", len(weights))

        resampled_samples = list(np.random.choice(paths_to_keep, size=dataset_size, replace=True, p=weights))

        with gzip.open(f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_hosp_icd_all.json.gz", "wt") as f:
            json.dump(resampled_samples, f)

    else:
        print("No weighting")
        dataset_size = min(dataset_size, len(paths_to_keep))
        resampled_samples = list(np.random.choice(paths_to_keep, size=dataset_size, replace=False))

        # Save the resampled samples with gzip compression
        with gzip.open(f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_hosp_icd_all.json.gz", "wt") as f:
            json.dump(resampled_samples, f)

    return resampled_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Load config for training")
    parser.add_argument('--task', type=str, default='pathway')
    return parser.parse_args()


if __name__ == '__main__':

    # get model path from command line arguments
    args = parse_args()
    task = args.task

    ''' For extract_event_counts to work, need to generate a subset of data without weighting before to count occurrences in. Run create_resampled_jsons() with weighting=False and dataset_size=1000000 first.

    # - Instructions for running the weighting script:
    #   - step1: extract_event_counts: calculate how often each column/event occurs in change_logs -> "{mimiciv_path}train_event_counts.json"
    #   - step2: calculate_weights: Given counts, calculate weight for each sample / timepoint -> "{mimiciv_path}{split}_sample_timepoint_paths.json.gz" and "..._sample_timepoint_weights" -> samples with many rare events are more important
    #   - step3: create_resampled_jsons: resample N timepoints based on weights for dataset size N -> if new dataset size is wanted, only this step needs to be rerun (very fast <1min)
    #
    #   Activate the desired method in the code; then run file'''

    if task == 'pathway':
        ''' standard training and eval jsons -> default training data '''
        create_resampled_jsons(split="train", dataset_size=1000000, weighting=True)
        create_resampled_jsons(split="val", weighting=False)

    elif task == 'ed_admission':
        ''' ed admission prediction '''
        stay_to_ids_dict = None
        create_resampled_jsons_ed_adm_only(split="train", weighting=True, dataset_size=1000000)
        create_resampled_jsons_ed_adm_only(split="val", weighting=False, dataset_size=5000)

    elif task == 'ed_icd':
        ''' ed icd prediction '''
        stay_to_ids_dict = None
        create_resampled_jsons_ed_icd_only(split="train", weighting=False, dataset_size=400000)
        create_resampled_jsons_ed_icd_only(split="val", weighting=False, dataset_size=5000)

    elif task == 'mort':
        ''' icu mortality prediction - start from weighted samples -> only filter for icu timepoints and balance by mortality label '''
        stay_to_ids_dict = None
        create_resampled_jsons_icu_mort(split="train")
        create_resampled_jsons_icu_mort(split="val")

    elif task == 'los':
        ''' icu LOS 3 day prediction - start from weighted samples and balance by LOS label '''
        stay_to_ids_dict = None
        create_resampled_jsons_icu_los_path(split="train")
        create_resampled_jsons_icu_los_path(split="val")

    elif task == 'hosp_icd':
        ''' hosp icd prediction '''
        stay_to_ids_dict = None
        create_resampled_jsons_hosp_icd(split="train", weighting=True, dataset_size=20000)
        create_resampled_jsons_hosp_icd(split="val", weighting=False, dataset_size=5000)

