import argparse
import gzip
import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from mimic_iv_extraction.paths import mimiciv_path as mimiciv_path
from patient_model.retrieve_change_log import retrieve_change_log_for_hour
from patient_model.retrieve_patient_model import retrieve_patient_model_, restrict_patient_until_hour, get_patient_description

def generate_patient_data_helper_from_timepoints_ed_adm(patient_stay_timepoints, add_los = True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = False):
    # add_los: if True, add hours till death or discharge or admission (ED) to the description and let it predict in next timepoint
    # val_set: if True, don't include los in the description, as it is not available in the validation set, but in the change log yes

    RESTRICT_TO_N_HOURS = RESTRICT_TO_N_HOURS
    SUMMARY_LEVEL = SUMMARY_LEVEL

    stay_id, timepoints = patient_stay_timepoints
    if initial_timepoint_only:
        timepoints = [timepoints[0]]
        assert timepoints[0] == 0, "Initial timepoint should be 0"

    complete_patient_stay = stay_to_ids_dict[stay_id]

    patient_folder = f"{mimiciv_path}all_data_ed_admission_pred{'_inital_timepoint' if initial_timepoint_only else '_all'}"

    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{patient_folder}_nohourrestriction"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    try:
        # if no ed stay for this patient, skip
        if 'ed_stay_hours' not in complete_patient_stay or complete_patient_stay['ed_stay_hours'] is None or complete_patient_stay['ed_stay_hours'] == 0:
            return
        if True or not os.path.exists(f"{patient_folder}/stay_{stay_id}.json"):
            samples = {}
            full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)
            final_disp = full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0]
            if final_disp not in ["ADMITTED", "HOME"]:
                return
            if final_disp == "HOME":
                final_disp = "DISCHARGED"

            for current_timepoint_idx in timepoints:

                current_hour_idx = current_timepoint_idx-1 # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0

                ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours']

                if not ed_stay: # here we only want ed_stay time_points
                    print(f"Skipping {patient_folder}/stay_{stay_id}.json, timepoint {current_timepoint_idx} is not in ED stay")
                    return

                prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay, restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

                desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=False, add_los=add_los and not val_set, los_only=RESTRICT_TO_N_HOURS==0) # diag_avail describes if ED diagnosis should be included as ED stay is over

                change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx, complete_patient_stay, stay_id, add_los=add_los, los_only=RESTRICT_TO_N_HOURS==0)

                if change_log is not None:
                    if initial_timepoint_only:
                        final_label_log = {'Emergency Department Stay': {'Future Disposition': final_disp}}
                        samples[current_timepoint_idx] = {'desc': desc, 'change_log': final_label_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}
                    else:
                        change_log['Emergency Department Stay']['Future Disposition'] = final_disp
                        samples[current_timepoint_idx] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}

            if len(samples) > 0:
                with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                    json.dump(samples, f)
        else:
            pass
    except Exception as e:
        print(f"Error processing stay {stay_id}: {e}")
        return

def generate_patient_data_helper_from_timepoints_ed_icd(patient_stay_timepoints, add_los = True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1):
    # add_los: if True, add hours till death or discharge or admission (ED) to the description and let it predict in next timepoint
    # val_set: if True, don't include los in the description, as it is not available in the validation set, but in the change log yes

    RESTRICT_TO_N_HOURS = RESTRICT_TO_N_HOURS
    SUMMARY_LEVEL = SUMMARY_LEVEL

    stay_id, timepoints = patient_stay_timepoints
    timepoints = [timepoints[0]] # duplicate timepoints possible because of resampling with replacement

    complete_patient_stay = stay_to_ids_dict[stay_id]

    patient_folder = f"{mimiciv_path}all_data_ed_icd_pred_all"

    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{patient_folder}_nohourrestriction"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    try:
        # if no ed stay for this patient, skip
        if 'ed_stay_hours' not in complete_patient_stay or complete_patient_stay['ed_stay_hours'] is None or complete_patient_stay['ed_stay_hours'] == 0:
            print(f"Skipping {patient_folder}/stay_{stay_id}.json")
            return
        if not os.path.exists(f"{patient_folder}/stay_{stay_id}.json"):
            samples = {}
            full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

            for current_timepoint_idx in timepoints:

                current_hour_idx = current_timepoint_idx-1 # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0

                ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours']

                if not ed_stay: # here we only want ed_stay time_points
                    print(f"Skipping {patient_folder}/stay_{stay_id}.json, timepoint {current_timepoint_idx} is not in ED stay")
                    return

                prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay, restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

                desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=False, add_los=add_los and not val_set, los_only=RESTRICT_TO_N_HOURS==0) # diag_avail describes if ED diagnosis should be included as ED stay is over


                change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx, complete_patient_stay, stay_id, add_los=add_los, los_only=RESTRICT_TO_N_HOURS==0)

                if change_log is not None:
                    samples[current_timepoint_idx] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}

            if len(samples) > 0:
                with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                    json.dump(samples, f)
        else:
            pass
    except Exception as e:
        print(f"Error processing stay {stay_id}: {e}")
        return

def generate_patient_data_helper_from_timepoints_icu_mort(patient_stay_timepoints, add_los = True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = False):
    # add_los: if True, add hours till death or discharge or admission (ED) to the description and let it predict in next timepoint
    # val_set: if True, don't include los in the description, as it is not available in the validation set, but in the change log yes
    # option 1: only last value of all variables
    # RESTRICT_TO_N_HOURS = None
    # SUMMARY_LEVEL = 3
    # option 2: all values in the last N hours
    RESTRICT_TO_N_HOURS = RESTRICT_TO_N_HOURS
    SUMMARY_LEVEL = SUMMARY_LEVEL
    # option 3: all values all time for summary model
    # RESTRICT_TO_N_HOURS = None
    # SUMMARY_LEVEL = 1

    stay_id, timepoints = patient_stay_timepoints
    complete_patient_stay = stay_to_ids_dict[stay_id]

    if 'icu_stay_ids' not in complete_patient_stay or len(complete_patient_stay['icu_stay_ids']) == 0:  # skip samples without ICU stay
        raise ValueError(f"Patient {stay_id} has no ICU stay")

    patient_folder = f"{mimiciv_path}all_data_icu_mort_pred{'_inital_timepoint' if initial_timepoint_only else '_all'}"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    try:
        if not os.path.exists(f"{patient_folder}/stay_{stay_id}.json"):
            samples = {}
            full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

            for current_timepoint_idx in timepoints:

                current_hour_idx = current_timepoint_idx-1 # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0

                ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours']+1
                if ed_stay:
                    raise ValueError(f"Patient {stay_id} is in ED stay at timepoint {current_timepoint_idx}")

                prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay, restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

                icu_outtime = None
                icu_idx = None
                for idx, stay in enumerate(full_patient_visit.patient_icu):
                    if stay.icustays.iloc[0]['intime'] <= hour_time <= stay.icustays.iloc[0]['outtime']:  # current stay -> save outtime
                        icu_outtime = stay.icustays.iloc[0]['outtime']
                        icu_idx = idx
                        break

                desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=True, add_los=add_los and not val_set, los_only=RESTRICT_TO_N_HOURS==0) # diag_avail describes if ED diagnosis should be included as ED stay is over

                change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx, complete_patient_stay, stay_id, add_los=add_los, los_only=RESTRICT_TO_N_HOURS==0)

                final_hosp_disposition = "DIED" if full_patient_visit.patient_adm.admissions.discharge_location.values[0] == "DIED" else "DISCHARGED"
                if final_hosp_disposition == "DIED":
                    # if died need to check if died within current ICU stay or not
                    time_of_death = full_patient_visit.patient_adm.admissions.deathtime.values[0]
                    if np.isnan(time_of_death):
                        return None
                    if icu_outtime is not None and time_of_death <= icu_outtime and time_of_death <= hour_time + pd.Timedelta(hours=24):  # death within next 24h and before ICU discharge
                        final_disp = "DIED"
                    else:
                        final_disp = "ALIVE"
                else:
                    final_disp = "ALIVE"

                if change_log is not None:
                    if initial_timepoint_only:
                        final_label_log = {'ICU Stay': {f'Stay {icu_idx}': {'24h Disposition': final_disp}}}
                        samples[current_timepoint_idx] = {'desc': desc, 'change_log': final_label_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}
                    else:
                        if 'ICU Stay' not in change_log:
                            change_log['ICU Stay'] = {}
                        if f'Stay {icu_idx}' not in change_log['ICU Stay']:
                            change_log['ICU Stay'][f'Stay {icu_idx}'] = {}
                        change_log['ICU Stay'][f'Stay {icu_idx}']['24h Disposition'] = final_disp
                        samples[current_timepoint_idx] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}

            if len(samples) > 0:
                with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                    json.dump(samples, f)
        else:
            print(f"Skipping {patient_folder}/stay_{stay_id}.json")
    except Exception as e:
        print(f"Error processing stay {stay_id}: {e}")
        return

def generate_patient_data_helper_from_timepoints_icu_los(patient_stay_timepoints, add_los = True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = False):
    # add_los: if True, add hours till death or discharge or admission (ED) to the description and let it predict in next timepoint
    # val_set: if True, don't include los in the description, as it is not available in the validation set, but in the change log yes

    RESTRICT_TO_N_HOURS = RESTRICT_TO_N_HOURS
    SUMMARY_LEVEL = SUMMARY_LEVEL

    stay_id, timepoints = patient_stay_timepoints
    complete_patient_stay = stay_to_ids_dict[stay_id]

    if 'icu_stay_ids' not in complete_patient_stay or len(complete_patient_stay['icu_stay_ids']) == 0:  # skip samples without ICU stay
        raise ValueError(f"Patient {stay_id} has no ICU stay")

    patient_folder = f"{mimiciv_path}all_data_icu_los_alltimepoints_pred{'_inital_timepoint' if initial_timepoint_only else '_all'}"

    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{patient_folder}_nohourrestriction"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    try:
        if not os.path.exists(f"{patient_folder}/stay_{stay_id}.json"):
            samples = {}
            full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

            for current_timepoint_idx in timepoints:

                current_hour_idx = current_timepoint_idx-1 # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0

                ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours']+1
                if ed_stay:
                    raise ValueError(f"Patient {stay_id} is in ED stay at timepoint {current_timepoint_idx}")

                prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay, restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

                icu_outtime = None
                icu_idx = None
                for idx, stay in enumerate(full_patient_visit.patient_icu):
                    if stay.icustays.iloc[0]['intime'] <= hour_time <= stay.icustays.iloc[0]['outtime']:  # current stay -> save outtime
                        icu_outtime = stay.icustays.iloc[0]['outtime']
                        icu_idx = idx
                        break

                desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=True, add_los=add_los and not val_set, los_only=RESTRICT_TO_N_HOURS==0) # diag_avail describes if ED diagnosis should be included as ED stay is over

                change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx, complete_patient_stay, stay_id, add_los=add_los, los_only=RESTRICT_TO_N_HOURS==0)

                time_to_release = icu_outtime - hour_time
                # LOS > 3 days
                if time_to_release > pd.Timedelta(days=3):
                    los_3day_label = 1
                else:
                    los_3day_label = 0
                final_disp = "YES" if los_3day_label else "NO"

                if change_log is not None:
                    if initial_timepoint_only:
                        final_label_log = {'ICU Stay': {f'Stay {icu_idx}': {'StayOver3days': final_disp}}} #3day LOS
                        samples[current_timepoint_idx] = {'desc': desc, 'change_log': final_label_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}
                    else:
                        if 'ICU Stay' not in change_log:
                            change_log['ICU Stay'] = {}
                        if f'Stay {icu_idx}' not in change_log['ICU Stay']:
                            change_log['ICU Stay'][f'Stay {icu_idx}'] = {}
                        change_log['ICU Stay'][f'Stay {icu_idx}']['StayOver3days'] = final_disp
                        samples[current_timepoint_idx] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}

            if len(samples) > 0:
                with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                    json.dump(samples, f)
        else:
            print(f"Skipping {patient_folder}/stay_{stay_id}.json")
    except Exception as e:
        print(f"Error processing stay {stay_id}: {e}")
        return

def generate_patient_data_helper_from_timepoints_hosp_icd(patient_stay_timepoints, add_los = True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1):
    # add_los: if True, add hours till death or discharge or admission (ED) to the description and let it predict in next timepoint
    # val_set: if True, don't include los in the description, as it is not available in the validation set, but in the change log yes
    RESTRICT_TO_N_HOURS = RESTRICT_TO_N_HOURS
    SUMMARY_LEVEL = SUMMARY_LEVEL

    stay_id, timepoints = patient_stay_timepoints
    timepoints = [timepoints[0]] # duplicate timepoints possible because of resampling with replacement

    complete_patient_stay = stay_to_ids_dict[stay_id]

    patient_folder = f"{mimiciv_path}all_data_hosp_icd_pred_all"

    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{patient_folder}_nohourrestriction"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    try:
        if not os.path.exists(f"{patient_folder}/stay_{stay_id}.json"):
            samples = {}
            full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

            for current_timepoint_idx in timepoints:

                current_hour_idx = current_timepoint_idx-1 # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0

                ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours']+1
                if ed_stay:
                    raise ValueError(f"Patient {stay_id} is in ED stay at timepoint {current_timepoint_idx}")

                prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay, restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

                desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=True, add_los=add_los and not val_set, los_only=RESTRICT_TO_N_HOURS==0) # diag_avail describes if ED diagnosis should be included as ED stay is over

                change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx, complete_patient_stay, stay_id, add_los=add_los, los_only=RESTRICT_TO_N_HOURS==0)

                if change_log is not None:
                    samples[current_timepoint_idx] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}

            if len(samples) > 0:
                with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                    json.dump(samples, f)
        else:
            print(f"Skipping {patient_folder}/stay_{stay_id}.json")
    except Exception as e:
        print(f"Error processing stay {stay_id}: {e}")
        return


''' pathway fine-tuning '''
''' 24  hour text model'''
def generate_patient_data_with_progress(data, initial_timepoint_only, task):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    if task == "ed_admission":
        result = generate_patient_data_helper_from_timepoints_ed_adm(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    elif task == "ed_icd":
        result = generate_patient_data_helper_from_timepoints_ed_icd(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1)
    elif task == "mort":
        result = generate_patient_data_helper_from_timepoints_icu_mort(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    elif task == "los":
        result = generate_patient_data_helper_from_timepoints_icu_los(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    elif task == "hosp_icd":
        result = generate_patient_data_helper_from_timepoints_hosp_icd(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1)
    # progress_bar.update(1)
    return result

def generate_patient_data_with_progress_val(data, initial_timepoint_only, task):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    if task == "ed_admission":
        result = generate_patient_data_helper_from_timepoints_ed_adm(data, add_los=True, val_set=True,  RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    elif task == "ed_icd":
        result = generate_patient_data_helper_from_timepoints_ed_icd(data, add_los=True, val_set=True, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1)
    elif task == "mort":
        result = generate_patient_data_helper_from_timepoints_icu_mort(data, add_los=True, val_set=True, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1,initial_timepoint_only=initial_timepoint_only)
    elif task == "los":
        result = generate_patient_data_helper_from_timepoints_icu_los(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    elif task == "hosp_icd":
        result = generate_patient_data_helper_from_timepoints_hosp_icd(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1)
    progress_bar.update(1)
    return result

''' for generating summary embeddings '''
def generate_patient_data_with_progress_summ(data, initial_timepoint_only, task):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    if task == "ed_admission":
        result = generate_patient_data_helper_from_timepoints_ed_adm(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    elif task == "ed_icd":
        result = generate_patient_data_helper_from_timepoints_ed_icd(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1)
    elif task == "los":
        result = generate_patient_data_helper_from_timepoints_icu_los(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    progress_bar.update(1)
    return result

def generate_patient_data_with_progress_val_summ(data, initial_timepoint_only, task):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    if task == "ed_admission":
        result = generate_patient_data_helper_from_timepoints_ed_adm(data, add_los=True, val_set=True,  RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    elif task == "ed_icd":
        result = generate_patient_data_helper_from_timepoints_ed_icd(data, add_los=True, val_set=True, RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1)
    elif task == "los":
        result = generate_patient_data_helper_from_timepoints_icu_los(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1, initial_timepoint_only = initial_timepoint_only)
    progress_bar.update(1)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Load config for training")
    parser.add_argument('--task', type=str, choices=["ed_admission", "ed_icd", "mort", "los", "hosp_icd"], required=True)
    parser.add_argument('--mode', type=str, choices=["path", "outcome"], required=True)
    parser.add_argument('--prepare', action='store_true', help="If set, only generate data needed for generating summary embeddings")
    return parser.parse_args()


if __name__ == '__main__':

    # get model path from command line arguments
    args = parse_args()
    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    start = time.time()

    task = args.task
    mode = args.mode

    if task == "ed_admission":

        ''' PREPARATION FOR FIRST TIMEPOINT / DIRECT PREDICTION '''
        if mode == "outcome":
            ''' for only first timepoint of each ED stay -> no weighting needed as all change_logs are same '''
            split = 'train'
            with gzip.open(f"{mimiciv_path}{split}_sample_timepoint_paths.json.gz", "rt") as f:
                data = json.load(f)

            split = 'val'
            with gzip.open(f"{mimiciv_path}{split}_sample_timepoint_paths.json.gz", "rt") as f:
                data_val = json.load(f)

            ''' after generating first hour only samples need to balance them and create train and val path json (usually they are generated in the weighting step) '''
            # timepoint_idx can always be 0 -> every sample has only one timepoint!
            files = os.listdir(f"{mimiciv_path}all_data_ed_admission_pred_inital_timepoint_nohourrestriction/")

            # filter out val files
            val_patients = set([elem.rsplit('_',1)[0] for elem in data_val])
            val_files = [f for f in files if f in val_patients]
            train_files = [f for f in files if f not in val_patients]

            # balance train and val paths based on label distribution
            train_labels = []
            val_labels = []
            for path in train_files:
                with open(f"{mimiciv_path}all_data_ed_admission_pred_inital_timepoint_nohourrestriction/{path}", "r") as f:
                    data = json.load(f)
                    label = data['0']['change_log']['Emergency Department Stay']['Future Disposition']
                    train_labels.append(0 if label == "HOME" else 1)

            for path in val_files:
                with open(f"{mimiciv_path}all_data_ed_admission_pred_inital_timepoint_nohourrestriction/{path}", "r") as f:
                    data = json.load(f)
                    label = data['0']['change_log']['Emergency Department Stay']['Future Disposition']
                    val_labels.append(0 if label == "HOME" else 1)

            path_to_label = dict(zip(val_files, val_labels))
            ones = [path for path in val_files if path_to_label[path] == 1]
            zeros = [path for path in val_files if path_to_label[path] == 0]
            minority_size = min(len(ones), len(zeros))
            val_balanced_samples = np.random.choice(ones, size=minority_size, replace=False).tolist() + \
                               np.random.choice(zeros, size=minority_size, replace=False).tolist()

            val_paths = [f"{f}_0" for f in val_balanced_samples]

            # balance train paths
            path_to_label = dict(zip(train_files, train_labels))
            ones = [path for path in train_files if path_to_label[path] == 1]
            zeros = [path for path in train_files if path_to_label[path] == 0]
            minority_size = min(len(ones), len(zeros))
            train_balanced_samples = np.random.choice(ones, size=minority_size, replace=False).tolist() + \
                                  np.random.choice(zeros, size=minority_size, replace=False).tolist()
            train_paths = [f"{f}_0" for f in train_balanced_samples]

            # save train and val paths
            with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz", "wt") as f:
                json.dump(train_paths, f)

            with gzip.open(f"{mimiciv_path}finetuning_data/val_noweighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz", "wt") as f:
                json.dump(val_paths, f)

        ''' for all timepoints of each ED stay -> weighting needed as all change_logs are different -> load pre-generated weighting files '''
        if mode == "path":
            with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_ed_admission_all.json.gz", "rt") as f:
                data = json.load(f)

            with gzip.open(f"{mimiciv_path}finetuning_data/val_noweighting_sample_timepoint_paths_ed_admission_all.json.gz", "rt") as f:
                data_val = json.load(f)

        elif mode == "outcome":
            with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz", "rt") as f:
                data = json.load(f)

            with gzip.open(f"{mimiciv_path}finetuning_data/val_noweighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz", "rt") as f:
                data_val = json.load(f)

    elif task == "ed_icd":
        # pathway fine-tuning == outcome fine-tuning
        with gzip.open(f"{mimiciv_path}finetuning_data/train_noweighting_sample_timepoint_paths_ed_icd_all.json.gz", "rt") as f:
            data = json.load(f)

        with gzip.open(f"{mimiciv_path}finetuning_data/val_noweighting_sample_timepoint_paths_ed_icd_all.json.gz", "rt") as f:
            data_val = json.load(f)

    elif task == "mort":
        #path and outcome use same data paths, as we train prediction at any timepoint for next 24 hours
        with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_icu_mort.json.gz","rt") as f:
            data = json.load(f)

        with gzip.open(f"{mimiciv_path}finetuning_data/val_weighting_sample_timepoint_paths_icu_mort.json.gz","rt") as f:
            data_val = json.load(f)

    elif task == "los":
        #path and outcome use same data paths, as we train prediction at any timepoint for next 24 hours
        with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_icu_los_path.json.gz", "rt") as f:
            data = json.load(f)

        with gzip.open(f"{mimiciv_path}finetuning_data/val_weighting_sample_timepoint_paths_icu_los_path.json.gz", "rt") as f:
            data_val = json.load(f)

    elif task == "hosp_icd":
        # pathway fine-tuning == outcome fine-tuning
        with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_hosp_icd_all.json.gz", "rt") as f:
            data = json.load(f)

        with gzip.open(f"{mimiciv_path}finetuning_data/val_noweighting_sample_timepoint_paths_hosp_icd_all.json.gz", "rt") as f:
            data_val = json.load(f)

    ''' shared code '''
    # convert this list of names such as 'stay_12345.json_97' to a list of tuples (stay_id, timepoint_idx)
    data = [t[5:].rsplit("_", 1) for t in data]
    data = [(t[0][:-5], t[1]) for t in data] #remove '.json' and convert timepoint_idx to int

    data_val = [t[5:].rsplit("_", 1) for t in data_val]
    data_val = [(t[0][:-5], t[1]) for t in data_val] #remove '.json' and convert timepoint_idx to int

    # group by stay_id
    stay_to_timepoints_dict = {}
    for stay_id, timepoint_idx in data:
        if stay_id in stay_to_timepoints_dict:
            stay_to_timepoints_dict[stay_id].append(int(timepoint_idx))
        else:
            stay_to_timepoints_dict[stay_id] = [int(timepoint_idx)]

    stay_to_timepoints_dict_val = {}
    for stay_id, timepoint_idx in data_val:
        if stay_id in stay_to_timepoints_dict_val:
            stay_to_timepoints_dict_val[stay_id].append(int(timepoint_idx))
        else:
            stay_to_timepoints_dict_val[stay_id] = [int(timepoint_idx)]

    n_workers = 200 #200
    chunksize = 10  # workload between samples is very different, therefore we want a rather small chunksize

    if mode == "path":
        # for text input
        partial_func = partial(generate_patient_data_with_progress, initial_timepoint_only=False, task=task)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_timepoints_dict)) as progress_bar:
                results = list(executor.map(partial_func, stay_to_timepoints_dict.items(), chunksize=chunksize))

        # val data
        partial_func = partial(generate_patient_data_with_progress_val, initial_timepoint_only=False, task=task)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_timepoints_dict_val)) as progress_bar:
                results = list(executor.map(partial_func, stay_to_timepoints_dict_val.items(), chunksize=chunksize))

        if task not in ['mort', 'los', 'hosp_icd']: #for mortality, hosp_icd and los-path only used already embedded samples
            # for summary embedding generation
            partial_func = partial(generate_patient_data_with_progress_summ, initial_timepoint_only=False, task=task)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                with tqdm(total=len(stay_to_timepoints_dict)) as progress_bar:
                    results = list(executor.map(partial_func, stay_to_timepoints_dict.items(), chunksize=chunksize))

            # # val data
            partial_func = partial(generate_patient_data_with_progress_val_summ, initial_timepoint_only=False, task=task)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                with tqdm(total=len(stay_to_timepoints_dict_val)) as progress_bar:
                    results = list(executor.map(partial_func, stay_to_timepoints_dict_val.items(), chunksize=chunksize))

    elif mode == "outcome":
        # for text input
        partial_func = partial(generate_patient_data_with_progress, initial_timepoint_only=True, task=task)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_timepoints_dict)) as progress_bar:
                results = list(executor.map(partial_func, stay_to_timepoints_dict.items(), chunksize=chunksize))

        # val data
        partial_func = partial(generate_patient_data_with_progress_val, initial_timepoint_only=True, task=task)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_timepoints_dict_val)) as progress_bar:
                results = list(executor.map(partial_func, stay_to_timepoints_dict_val.items(), chunksize=chunksize))

