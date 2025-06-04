import argparse
import gzip
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

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

def generate_sample_description(complete_patient_stay, full_patient_visit, prior_patient_state, hour_idx, hour_time, SUMMARY_LEVEL=1, add_los=False, los_only=False, ed_stay=False):

    desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL,
                                   diag_avail=not ed_stay, add_los=add_los, los_only=los_only, ed_stay=ed_stay)
    # for summary embeddings we don't include "DISCHARGED FROM ICU" or "ADMITTED TO ICU" into the description as it is not relevant for the summary but included in text (always about last hour)

    return desc


# only needed for first data generation before weights are calculated
def generate_patient_data_helper(patient_stay):
    RESTRICT_TO_N_HOURS = 24 #hour window to consider before the current hour
    SUMMARY_LEVEL = 1

    stay_id, complete_patient_stay = patient_stay
    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{mimiciv_path}all_data_unweighted"
    else:
        patient_folder = f"{mimiciv_path}all_data_{RESTRICT_TO_N_HOURS}_hours_unweighted"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    try:
        if not os.path.exists(f"{patient_folder}/stay_{stay_id}.json"):
            samples = []
            full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)
            current_hour_idx = -1
            end_reached = False
            just_admitted = False
            entered_icu = False
            released_from_icu_idx = -1
            ed_stay = full_patient_visit.patient_ed is not None #check if patient was in ED or directly admitted to Hospital
            while not end_reached:
                prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay, restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour
                desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=(not ed_stay) or just_admitted) # diag_avail describes if ED diagnosis should be included as ED stay is over
                if just_admitted:
                    desc['Emergency Department Stay']['Disposition'] = 'ADMITTED'
                    just_admitted = False
                    ed_stay = False # patient is now admitted to Hospital

                if released_from_icu_idx != -1:
                    if 'Hospital Stay' in desc:# patient is released from ICU this timepoint -> was mentioned in previous change log
                        desc['Hospital Stay']['Disposition'] = 'DISCHARGED FROM ICU'

                if entered_icu: # patient is admitted to ICU this timepoint -> was mentioned in previous change log -> if patient was released and admitted at same time, should again say admitted
                    if 'Hospital Stay' in desc: # if not, patient still in ED, but hospital stay will be added in next timepoint
                        desc['Hospital Stay']['Disposition'] = 'ADMITTED TO ICU'

                change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx, complete_patient_stay, stay_id)

                if change_log is not None:
                    samples.append({'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx})

                current_hour_idx += 1
                if next_patient_state is None:
                    if change_log is not None and 'Emergency Department Stay' in change_log and change_log['Emergency Department Stay']['Disposition'] == 'ADMITTED' and full_patient_visit.patient_adm is not None:
                        just_admitted = True
                    else:
                        end_reached = True

            if len(samples) > 0:
                with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                    json.dump(samples, f)

        else:
            print(f"Skipping {patient_folder}/stay_{stay_id}.json")
    except Exception as e:
        print(f"Error processing stay {stay_id}: {e}")
        return

def generate_patient_data_helper_from_timepoints(patient_stay_timepoints, add_los = True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1):
    # add_los: if True, add hours till death or discharge or admission (ED) to the description and let it predict in next timepoint
    # val_set: if True, don't include los in the description, as it is not available in the validation set, but in the change log yes

    RESTRICT_TO_N_HOURS = RESTRICT_TO_N_HOURS
    SUMMARY_LEVEL = SUMMARY_LEVEL

    stay_id, timepoints = patient_stay_timepoints
    complete_patient_stay = stay_to_ids_dict[stay_id]

    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{mimiciv_path}all_data_summarylvl1"
    elif RESTRICT_TO_N_HOURS == 0:
        patient_folder = f"{mimiciv_path}all_data_losonly"
    else:
        patient_folder = f"{mimiciv_path}all_data_{RESTRICT_TO_N_HOURS}_hours"

    if add_los:
        patient_folder = patient_folder + "_los_noisy"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    try:
        if not os.path.exists(f"{patient_folder}/stay_{stay_id}.json"):
            samples = {}
            full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

            for current_timepoint_idx in timepoints:

                current_hour_idx = current_timepoint_idx-1 # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0

                ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours']+1
                if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] != 0 and current_hour_idx == complete_patient_stay['ed_stay_hours'] + 1:
                    if full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0] == "ADMITTED" or (complete_patient_stay['hosp_stay_hours'] > 0 and full_patient_visit.patient_adm.dischtime > full_patient_visit.patient_ed.outtime):
                        just_admitted = True
                    else:
                        just_admitted = False
                else:
                    just_admitted = False

                prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay, restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

                entered_icu = False
                released_from_icu_idx = -1
                if 'icu_stay_ids' in complete_patient_stay and len(complete_patient_stay['icu_stay_ids']) > 0:
                    for idx, stay in enumerate(full_patient_visit.patient_icu):
                        if stay.icustays.iloc[0]['intime'] == hour_time:
                            entered_icu = True

                        if stay.icustays.iloc[0]['outtime'] == hour_time:
                            released_from_icu_idx = idx

                desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=(not ed_stay) or just_admitted, add_los=add_los and not val_set, los_only=RESTRICT_TO_N_HOURS==0) # diag_avail describes if ED diagnosis should be included as ED stay is over
                if just_admitted:
                    desc['Emergency Department Stay']['Disposition'] = 'ADMITTED'

                if released_from_icu_idx != -1:
                    if 'Hospital Stay' in desc:# patient is released from ICU this timepoint -> was mentioned in previous change log
                        desc['Hospital Stay']['Disposition'] = 'DISCHARGED FROM ICU'

                if entered_icu: # patient is admitted to ICU this timepoint -> was mentioned in previous change log -> if patient was released and admitted at same time, should again say admitted
                    if 'Hospital Stay' in desc: # if not, patient still in ED, but hospital stay will be added in next timepoint
                        desc['Hospital Stay']['Disposition'] = 'ADMITTED TO ICU'

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

''' 24  hour text model'''
def generate_patient_data_with_progress(data):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    result = generate_patient_data_helper_from_timepoints(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1)
    progress_bar.update(1)
    return result

def generate_patient_data_with_progress_val(data):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    result = generate_patient_data_helper_from_timepoints(data, add_los=True, val_set=True,  RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1)
    progress_bar.update(1)
    return result

''' summary model (for training summarizer model) '''
def generate_patient_data_with_progress_summ(data):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    result = generate_patient_data_helper_from_timepoints(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1)
    progress_bar.update(1)
    return result

def generate_patient_data_with_progress_val_summ(data):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    result = generate_patient_data_helper_from_timepoints(data, add_los=True, val_set=True,  RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1)
    progress_bar.update(1)
    return result

''' LOS only model -> for training summary only model '''
def generate_patient_data_with_progress_Los_only(data):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    result = generate_patient_data_helper_from_timepoints(data, add_los=True, val_set=False, RESTRICT_TO_N_HOURS=0, SUMMARY_LEVEL=1)
    progress_bar.update(1)
    return result

def generate_patient_data_with_progress_val_Los_only(data):
    # result = generate_patient_data_helper(data) # needed for first data generation to create weights etc later
    result = generate_patient_data_helper_from_timepoints(data, add_los=True, val_set=True,  RESTRICT_TO_N_HOURS=0, SUMMARY_LEVEL=1)
    progress_bar.update(1)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Load config for generating dataset")
    parser.add_argument('--mode', type=str, default='text_model', choices=['text_model', 'summary_model', 'summary_model_LOS'],)
    return parser.parse_args()


if __name__ == '__main__':

    ''' Warning: these methods are quite I/O intensive, so it is recommended to run this on a machine with SSDs and enough RAM. '''

    # get model path from command line arguments
    args = parse_args()
    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    with gzip.open(f"{mimiciv_path}train_weighting_sample_timepoint_paths_1Mio.json.gz", "rt") as f: # training data for 1 Mio sample WITH WEIGHTING
        data_train = json.load(f)

    with gzip.open(f"{mimiciv_path}val_noweighting_sample_timepoint_paths.json.gz", "rt") as f: # standard val data (no weighting)
        data_val = json.load(f)

    with gzip.open(f"{mimiciv_path}test_noweighting_sample_timepoint_paths.json.gz", "rt") as f:
        data_test = json.load(f)

    # convert this list of names such as 'stay_12345.json_97' to a list of tuples (stay_id, timepoint_idx)
    data_train = [t[5:].rsplit("_", 1) for t in data_train]
    data_train = [(t[0][:-5], t[1]) for t in data_train] #remove '.json' and convert timepoint_idx to int

    data_val = [t[5:].rsplit("_", 1) for t in data_val]
    data_val = [(t[0][:-5], t[1]) for t in data_val] #remove '.json' and convert timepoint_idx to int

    data_test = [t[5:].rsplit("_", 1) for t in data_test]
    data_test = [(t[0][:-5], t[1]) for t in data_test] #remove '.json' and convert timepoint_idx to int

    # group by stay_id
    stay_to_timepoints_dict = {}
    for stay_id, timepoint_idx in data_train:
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

    stay_to_timepoints_dict_test = {}
    for stay_id, timepoint_idx in data_test:
        if stay_id in stay_to_timepoints_dict_test:
            stay_to_timepoints_dict_test[stay_id].append(int(timepoint_idx))
        else:
            stay_to_timepoints_dict_test[stay_id] = [int(timepoint_idx)]

    n_workers = 200 #200 #set according to your machine, can be smaller if you have less RAM or CPU cores
    chunksize = 10  # workload between samples is very different, therefore we want a rather small chunksize


    if args.mode == 'text_model':

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress, stay_to_timepoints_dict.items(), chunksize=chunksize))

        # val data
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_val, stay_to_timepoints_dict_val.items(), chunksize=chunksize))

        # test data
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_val, stay_to_timepoints_dict_test.items(), chunksize=chunksize))


    elif args.mode == 'summary_model':

        # Initialize the progress bar
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_summ, stay_to_timepoints_dict.items(), chunksize=chunksize))

        # val data
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_val_summ, stay_to_timepoints_dict_val.items(), chunksize=chunksize))

        # test data
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_val_summ, stay_to_timepoints_dict_test.items(), chunksize=chunksize))



    elif args.mode == 'summary_model_LOS': #for the summary model, we still need text representation for the LOS indicator, this is generated here.
        # Initialize the progress bar
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_Los_only, stay_to_timepoints_dict.items(), chunksize=chunksize))

        # val data
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_val_Los_only, stay_to_timepoints_dict_val.items(), chunksize=chunksize))

        # test data
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(stay_to_ids_dict)) as progress_bar:
                results = list(executor.map(generate_patient_data_with_progress_val_Los_only, stay_to_timepoints_dict_test.items(), chunksize=chunksize))


    # Make sure to close the progress bar
    progress_bar.close()