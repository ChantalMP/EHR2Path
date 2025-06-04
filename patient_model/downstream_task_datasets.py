import argparse
import json
import json
import os
import random

import numpy as np
import pandas as pd
from datasets import tqdm
from torch.utils.data import Dataset

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from mimic_iv_extraction.paths import mimiciv_path
from patient_model.retrieve_change_log import retrieve_change_log_for_hour
from patient_model.retrieve_patient_model import retrieve_patient_model_, restrict_patient_until_hour, \
    get_patient_description


'''
Create a val/test dataset containing all hospital patient stays at time of admission -> includes ED stay data if available and all static information about patient
'''

def extract_ED_data(split="val", at_discharge=False, use_los=False, at_admission=True):
    # RESTRICT_TO_N_HOURS = None
    # SUMMARY_LEVEL = 1
    RESTRICT_TO_N_HOURS = 24
    SUMMARY_LEVEL = 1

    if at_admission: #all_data_24_hours_ED_adm
        if RESTRICT_TO_N_HOURS is None:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_ED_adm_{split}"
        else:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_ED_adm_{split}"
    else:
        if RESTRICT_TO_N_HOURS is None:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_ED_{split}"
        else:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_ED_{split}"

    if at_discharge:
        patient_folder = patient_folder + "_at_discharge"

    if use_los:
        patient_folder = patient_folder + "_los"

    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    val_patient_paths = json.load(open(f"{mimiciv_path}{split}_paths.json", "r"))

    ED_timepoint_paths = []
    ED_adm_labels = []

    for idx, patient_path in tqdm(enumerate(val_patient_paths)):
        stay_id = patient_path.split('/')[-1][5:-5]
        complete_patient_stay = stay_to_ids_dict[stay_id]
        if 'ed_stay_hours' not in complete_patient_stay or complete_patient_stay['ed_stay_hours'] == None or complete_patient_stay[
            'ed_stay_hours'] <= 0:  # skip samples without ED stay
            continue

        samples = {}
        full_patient_visit, _ = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # check if discharge location is available (or nan) as well as patients that died in ED
        # possible disposition values: HOME, ADMITTED, TRANSFER, LEFT WITHOUT BEING SEEN, OTHER, LEFT AGAINST MEDICAL ADVICE, ELOPED, EXPIRED
        # for Admission prediction task we only consider patients that were admitted or discharged (=HOME)
        if full_patient_visit.patient_ed is None:
            continue
        disposition = full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0]
        if at_admission and disposition not in ["ADMITTED", "HOME"]:
            continue

        # extract all admission timepoints
        if at_admission:
            target_timepoint_idx = 0
        elif at_discharge:
            target_timepoint_idx = complete_patient_stay['ed_stay_hours'] + 1
        else:
            # choose random timepoint within ED stay
            target_timepoint_idx = random.randint(0, complete_patient_stay['ed_stay_hours'])

        # generate samples for all timepoints
        current_hour_idx = int(target_timepoint_idx) - 1
        ed_stay = True

        prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay,
                                                                     restrict_to_N_hours=RESTRICT_TO_N_HOURS)

        desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=False,
                                       add_los=use_los, force_los=use_los)

        change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx,
                                                                                                          complete_patient_stay, stay_id,
                                                                                                          add_los=use_los)

        if change_log is not None:
            samples[int(target_timepoint_idx)] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}
            ED_timepoint_paths.append(f"stay_{stay_id}.json_{int(target_timepoint_idx)}")
            ED_adm_labels.append(1 if disposition == "ADMITTED" else 0)

        if len(samples) > 0:
            with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                json.dump(samples, f)

    if at_discharge:
        with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_at_discharge_los_timepoint_paths.json", "w") as f:
            json.dump(ED_timepoint_paths, f)
    elif at_admission:
        with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_timepoint_paths.json", "w") as f:
            json.dump(ED_timepoint_paths, f)
        with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_labels.json", "w") as f:
            json.dump(ED_adm_labels, f)
    else:
        with open(f"{mimiciv_path}eval_tasks/{split}_ED_timepoint_paths.json", "w") as f:
            json.dump(ED_timepoint_paths, f)

class EDDataset(Dataset):
    def __init__(self, tokenizer, split='val', task="ED_Adm", at_discharge=False, use_los=False, predict_adm = False, eval_future_ed_disp=False):
        self.task = task

        if at_discharge:
            if use_los:
                with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_at_discharge_los_timepoint_paths.json", "r") as f:
                    self.data = json.load(f)
            else:
                with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_at_discharge_timepoint_paths.json", "r") as f:
                    self.data = json.load(f)
        else:
            with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_timepoint_paths.json", "r") as f:
                self.data = json.load(f)

        with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_labels.json", "r") as f:
            self.ed_adm_labels = json.load(f)

        print(f"Loaded {len(self.data)} samples for {split} split")

        if not predict_adm: #predict ICD -> just use 500 samples
            if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_ED_icd_sample_final_ids.json"):
                with open(f"{mimiciv_path}eval_tasks/{split}_ED_icd_sample_final_ids.json", "r") as f:
                    self.data = json.load(f)

            else:
                rng = np.random.RandomState(42)
                # sample random 500 samples
                self.data = rng.choice(self.data, size=500, replace=False)

                # save sample idxs for publishing val and test sets
                with open(f"{mimiciv_path}eval_tasks/{split}_ED_icd_sample_final_ids.json", "w") as f:
                    json.dump(list(self.data), f)

        else: # sample balanced between admitted and not admitted patients
            if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_ED_adm_sample_final_ids.json"):
                with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_sample_final_ids.json", "r") as f:
                    self.data = json.load(f)

                if not eval_future_ed_disp and split == 'val':
                    # only take 100 pos and 100 neg samples
                    self.data = self.data[:100] + self.data[-100:]

            else:
                # fix seed for reproducibility
                rng = np.random.RandomState(42)
                pos_idxs = list(rng.choice([idx for idx, label in enumerate(self.ed_adm_labels) if label == 1], size=250, replace=False))
                neg_idxs = list(rng.choice([idx for idx, label in enumerate(self.ed_adm_labels) if label == 0], size=250, replace=False))

                self.data = [self.data[idx] for idx in pos_idxs + neg_idxs]

                # save sample idxs for publishing val and test sets
                with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_sample_final_ids.json", "w") as f:
                    json.dump(list(self.data), f)

        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        self.tokenizer = tokenizer
        self.split = split

        self.at_discharge = at_discharge
        self.use_los = use_los

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        if self.at_discharge:
            if self.use_los:
                full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ED_{self.split}_at_discharge_los/" + sample_path, "r"))
            else:
                full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ED_{self.split}_at_discharge/" + sample_path, "r"))
        else:
            full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ED_adm_{self.split}/" + sample_path, "r"))
        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]
        stay_id = sample["stay_id"]

        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # Hosp dataset, so these are always False
        ed_stay = True

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=ed_stay)

        # 'ED_Adm'
        final_ed_disposition = "ADMITTED" if full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0] == "ADMITTED" else "DISCHARGED"
        label = {'ed_adm': 1 if final_ed_disposition == "ADMITTED" else 0}

        # ED ICD
        icd_codes = full_patient_visit.patient_ed.icd_categories.split(";")
        label['icd'] = icd_codes

        return {
            'label': label,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': ed_stay,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }

class EDTimeseriesDataset(Dataset):
    def __init__(self, tokenizer, split='val', value_cols=["vitalsigns"], time_tolerance=1, at_admission=True):
        if at_admission:
            with open(f"{mimiciv_path}eval_tasks/{split}_ED_adm_timepoint_paths.json", "r") as f:
                self.data = json.load(f)
        else:
            with open(f"{mimiciv_path}eval_tasks/{split}_ED_timepoint_paths.json", "r") as f:
                self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples for {split} split")

        if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_ed_ts_sample_final_ids.json"):
            with open(f"{mimiciv_path}eval_tasks/{split}_ed_ts_sample_final_ids.json", "r") as f:
                self.data = json.load(f)

        else:
            rng = np.random.RandomState(42)
            self.data = rng.choice(self.data, size=500, replace=False)

            with open(f"{mimiciv_path}eval_tasks/{split}_ed_ts_sample_final_ids.json", "w") as f:
                json.dump(list(self.data), f)

        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        self.tokenizer = tokenizer
        self.split = split
        self.value_cols = value_cols  # vitalsigns, medications
        self.time_tolerance = time_tolerance
        self.at_admission = at_admission

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        if self.at_admission:
            full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ED_adm_{self.split}/" + sample_path, "r"))
        else:
            full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ED_{self.split}/" + sample_path, "r"))
        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]
        stay_id = sample["stay_id"]

        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # Hosp dataset, so these are always False
        ed_stay = True

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=ed_stay)

        end_hour_idx = hour_idx + 24  # tolerance hours extra to compute time-tolerant metrics at early and late timepoints as well
        relevant_time_window, end_time = restrict_patient_until_hour(full_patient_visit, end_hour_idx, ed_stay=ed_stay, restrict_to_N_hours=24)
        relevant_time_window_time_tolerant, end_time_ = restrict_patient_until_hour(full_patient_visit, end_hour_idx + self.time_tolerance,
                                                                                   ed_stay=ed_stay, restrict_to_N_hours=24 + self.time_tolerance)

        label = {}
        label_time_tolerant = {}
        if "vitalsigns" in self.value_cols:
            label["vitalsigns"] = relevant_time_window.patient_ed.ed_vital
            label_time_tolerant["vitalsigns"] = relevant_time_window_time_tolerant.patient_ed.ed_vital

        if "medications" in self.value_cols:
            label["medications"] = relevant_time_window.patient_ed.ed_pyxis
            label_time_tolerant["medications"] = relevant_time_window_time_tolerant.patient_ed.ed_pyxis

        else:
            raise ValueError(f"Value column {self.value_col} not supported")

        return {
            'label': label,
            'label_time_tolerant': label_time_tolerant,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': ed_stay,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }

def extract_last_Hosp_data(split="val", use_los=False):
    # RESTRICT_TO_N_HOURS = None
    # SUMMARY_LEVEL = 1
    RESTRICT_TO_N_HOURS = 24
    SUMMARY_LEVEL = 1

    if RESTRICT_TO_N_HOURS is None:
        if use_los:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_DISCH_{split}_los"
        else:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_DISCH_{split}"
    else:
        if use_los:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_DISCH_{split}_los"
        else:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_DISCH_{split}"
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    val_patient_paths = json.load(open(f"{mimiciv_path}{split}_paths.json", "r"))

    ADM_timepoint_paths = []

    for idx, patient_path in tqdm(enumerate(val_patient_paths)):
        stay_id = patient_path.split('/')[-1][5:-5]
        complete_patient_stay = stay_to_ids_dict[stay_id]
        if 'hosp_stay_hours' not in complete_patient_stay or complete_patient_stay['hosp_stay_hours'] == None or complete_patient_stay[
            'hosp_stay_hours'] <= 0:  # skip samples without Hosp stay
            continue

        samples = {}
        full_patient_visit, _ = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # check if discharge location is available (or nan)
        if full_patient_visit.patient_adm.admissions.discharge_location.isnull().values.any():
            continue

        # extract all admission timepoints
        target_timepoint_idx = complete_patient_stay['hosp_stay_hours'] + 1
        # if ed_stay need to add ed_stay_length to hour_idx
        if 'ed_stay_hours' in complete_patient_stay and full_patient_visit.patient_ed is not None:
            target_timepoint_idx += complete_patient_stay['ed_stay_hours'] + 2  # +2 for transition idxs

        # generate samples for all timepoints
        current_hour_idx = int(target_timepoint_idx) - 1
        ed_stay = False

        if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] != 0 and current_hour_idx == complete_patient_stay[
            'ed_stay_hours'] + 1:
            if full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0] == "ADMITTED" or (complete_patient_stay[
                                                                                                   'hosp_stay_hours'] > 0 and full_patient_visit.patient_adm.dischtime > full_patient_visit.patient_ed.outtime):
                just_admitted = True
            else:
                just_admitted = False  # patient officially in hospital, but actually stays in ED
                continue
        else:
            just_admitted = False

        prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay,
                                                                     restrict_to_N_hours=RESTRICT_TO_N_HOURS)

        desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=(not ed_stay) or just_admitted, add_los=use_los, force_los=True)  # diag_avail describes if ED diagnosis should be included as ED stay is over

        change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx,
                                                                                                          complete_patient_stay, stay_id,
                                                                                                          add_los=use_los)

        if change_log is not None:
            samples[int(target_timepoint_idx)] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}
            ADM_timepoint_paths.append(f"stay_{stay_id}.json_{int(target_timepoint_idx)}")

        if len(samples) > 0:
            with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                json.dump(samples, f)

    if use_los:
        with open(f"{mimiciv_path}eval_tasks/{split}_DISCH_los_timepoint_paths.json", "w") as f:
            json.dump(ADM_timepoint_paths, f)
    else:
        with open(f"{mimiciv_path}eval_tasks/{split}_DISCH_timepoint_paths.json", "w") as f:
            json.dump(ADM_timepoint_paths, f)
class AdmissionDataset(Dataset):
    def __init__(self, tokenizer, split='val', task="Adm_Mort"):
        self.task = task

        with open(f"{mimiciv_path}eval_tasks/{split}_ADM_timepoint_paths.json", "r") as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples for {split} split")

        self.data = self.data[:500]
        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        self.tokenizer = tokenizer
        self.split = split

        # self.data = self.data[:5]

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ADM_{self.split}/" + sample_path, "r"))
        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]
        stay_id = sample["stay_id"]

        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # Hosp dataset, so these are always False
        ed_stay = False

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=ed_stay)
        assert prior_time < prior_patient_state.patient_adm.admittime, f"Prior time {prior_time} should be before admittime {prior_patient_state.patient_adm.admittime}"

        if self.task == 'Adm_Mort':
            final_hosp_disposition = "DIED" if full_patient_visit.patient_adm.admissions.discharge_location.values[0] == "DIED" else "DISCHARGED"
            label = {'adm_mort': 1 if final_hosp_disposition == "DIED" else 0}

        else:
            raise ValueError(f"Task {self.task} not supported")

        return {
            'label': label,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': ed_stay,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }

class DischargeDataset(Dataset):
    def __init__(self, tokenizer, split='val', task="Disch_ICD", use_los=False):
        self.task = task

        if use_los:
            with open(f"{mimiciv_path}eval_tasks/{split}_DISCH_los_timepoint_paths.json", "r") as f:
                self.data = json.load(f)
        else:
            with open(f"{mimiciv_path}eval_tasks/{split}_DISCH_timepoint_paths.json", "r") as f:
                self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples for {split} split")

        if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_DISCH_sample_final_ids.json"):
            with open(f"{mimiciv_path}eval_tasks/{split}_DISCH_sample_final_ids.json", "r") as f:
                self.data = json.load(f)

        else:
            rng = np.random.RandomState(42)
            self.data = rng.choice(self.data, size=500, replace=False)
            with open(f"{mimiciv_path}eval_tasks/{split}_DISCH_sample_final_ids.json", "w") as f:
                json.dump(list(self.data), f)

        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        self.tokenizer = tokenizer
        self.split = split
        self.use_los = use_los

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        if self.use_los:
            full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_DISCH_{self.split}_los/" + sample_path, "r"))
        else:
            full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_DISCH_{self.split}/" + sample_path, "r"))
        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]
        stay_id = sample["stay_id"]

        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # Hosp dataset, so these are always False
        ed_stay = False

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=ed_stay)

        icd_codes = full_patient_visit.patient_adm.icd_categories.split(";")
        label = {'icd': icd_codes}

        return {
            'label': label,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': ed_stay,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }


'''
Create a val/test dataset containing all ICU patient stays at 24h after ICU admission for LOS > 3 days task
'''
def extract_ICU_val_data_24h(split="val", use_los=False):
    # RESTRICT_TO_N_HOURS = None
    # SUMMARY_LEVEL = 1
    RESTRICT_TO_N_HOURS = 24
    SUMMARY_LEVEL = 1

    if RESTRICT_TO_N_HOURS is None:
        if use_los:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_ICU24_{split}_los"
        else:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_ICU24_{split}"
    else:
        if use_los:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_ICU24_{split}_los"
        else:
            patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_ICU24_{split}"
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    val_patient_paths = json.load(open(f"{mimiciv_path}{split}_paths.json", "r"))

    ICU24_timepoint_paths = []
    ICU24_los_3day_labels = []

    processed_stays = [f[5:-5] for f in os.listdir(patient_folder)]

    for patient_path in tqdm(val_patient_paths):
        stay_id = patient_path.split('/')[-1][5:-5]
        if stay_id in processed_stays:
            continue
        complete_patient_stay = stay_to_ids_dict[stay_id]
        if 'icu_stay_ids' not in complete_patient_stay or len(complete_patient_stay['icu_stay_ids']) == 0:  # skip samples without ICU stay
            continue

        samples = {}
        full_patient_visit, _ = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # extract all timepoints that are 24h after ICU admission for all ICU stays of the patient
        target_timepoints = []
        for idx, stay in enumerate(full_patient_visit.patient_icu):
            start_time = stay.icustays.iloc[0]['intime']
            end_time = stay.icustays.iloc[0]['outtime']
            if end_time - start_time <= pd.Timedelta(hours=24):  # skip stays shorter than 24h in ICU
                continue
            target_time = start_time + pd.Timedelta(hours=24)
            target_timepoint_idx = (target_time - full_patient_visit.patient_adm.admittime)
            target_timepoint_idx = target_timepoint_idx.total_seconds() / 3600 + 1  # timepoint idx should start with 1 at admission (0 is timepoint directly before admission)
            # if ed_stay need to add ed_stay_length to hour_idx
            if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] != 0:
                target_timepoint_idx += complete_patient_stay['ed_stay_hours'] + 2  # +2 for transition idxs
            target_timepoints.append(target_timepoint_idx)

        # generate samples for all timepoints
        for current_timepoint_idx in target_timepoints:
            current_hour_idx = int(current_timepoint_idx) - 1
            ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours'] + 1
            assert ed_stay == False, f"ED stay should be over at this point {stay_id}"
            if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] != 0 and current_hour_idx == complete_patient_stay[
                'ed_stay_hours'] + 1:
                if full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0] == "ADMITTED" or (complete_patient_stay[
                                                                                                       'hosp_stay_hours'] > 0 and full_patient_visit.patient_adm.dischtime > full_patient_visit.patient_ed.outtime):
                    raise ValueError(f"Entered ICU should not happen for 24h ICU stays {stay_id}")
                else:
                    just_admitted = False
            else:
                just_admitted = False

            prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay,
                                                                         restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

            icu_outtime = None
            for idx, stay in enumerate(full_patient_visit.patient_icu):
                if stay.icustays.iloc[0]['intime'] == hour_time:
                    print(f"Entered ICU should not happen for 24h ICU stays {stay_id}")
                    raise ValueError(f"Entered ICU should not happen for 24h ICU stays {stay_id}")

                if stay.icustays.iloc[0]['outtime'] == hour_time:
                    print(f"Released from ICU should not happen for 24h ICU stays {stay_id}")
                    raise ValueError(f"Released from ICU should not happen for 24h ICU stays {stay_id}")

                if stay.icustays.iloc[0]['intime'] <= hour_time <= stay.icustays.iloc[0]['outtime']:  # current stay -> save outtime
                    icu_outtime = stay.icustays.iloc[0]['outtime']
                    break

            if icu_outtime is None:
                continue

            desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=(
                                                                                                                                                     not ed_stay) or just_admitted,
                                           add_los=use_los)  # diag_avail describes if ED diagnosis should be included as ED stay is over

            change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx,
                                                                                                              complete_patient_stay, stay_id,
                                                                                                              add_los=use_los)

            time_to_release = icu_outtime - hour_time
            # LOS > 3 days
            if time_to_release > pd.Timedelta(days=3):
                los_3day_label = 1
            else:
                los_3day_label = 0

            if change_log is not None:
                samples[int(current_timepoint_idx)] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx,
                                                       'los_3day_label': los_3day_label}
                ICU24_timepoint_paths.append(f"stay_{stay_id}.json_{int(current_timepoint_idx)}")
                ICU24_los_3day_labels.append(los_3day_label)

        if len(samples) > 0:
            with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                json.dump(samples, f)

    if use_los:
        with open(f"{mimiciv_path}eval_tasks/{split}_ICU24_los_timepoint_paths.json", "w") as f:
            json.dump(ICU24_timepoint_paths, f)
    else:
        with open(f"{mimiciv_path}eval_tasks/{split}_ICU24_timepoint_paths.json", "w") as f:
            json.dump(ICU24_timepoint_paths, f)

    with open(f"{mimiciv_path}eval_tasks/{split}_ICU24_los_3day_labels.json", "w") as f:
        json.dump(ICU24_los_3day_labels, f)


'''
Includes all ICU stays at hour 24 after ICU admission
Labels for ICU LOS > 3 days
'''


class ICU24hDataset(Dataset):
    def __init__(self, tokenizer, split='val', use_los=False, eval_future_los=False):

        if use_los:
            with open(f"{mimiciv_path}eval_tasks/{split}_ICU24_los_timepoint_paths.json", "r") as f:
                self.data = json.load(f)
        else:
            with open(f"{mimiciv_path}eval_tasks/{split}_ICU24_timepoint_paths.json", "r") as f:
                self.data = json.load(f)

        # load labels
        with open(f"{mimiciv_path}eval_tasks/{split}_ICU24_los_3day_labels.json", "r") as f:
            self.los_3day_labels = json.load(f)

        print(f"Loaded {len(self.data)} samples for {split} split")

        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_icu_los_sample_final_ids.json"):
            with open(f"{mimiciv_path}eval_tasks/{split}_icu_los_sample_final_ids.json", "r") as f:
                self.data = json.load(f)

            if not eval_future_los and split == 'val':
                # only take 100 pos and 100 neg samples
                self.data = self.data[:100] + self.data[-100:]

        else:
            # sample 500 samples, 50% positive given self.los_3day_labels
            # fix seed for reproducibility
            rng = np.random.RandomState(42)
            pos_idxs = [idx for idx, label in enumerate(self.los_3day_labels) if label == 1]
            pos_idxs = list(rng.choice(pos_idxs, size=min(len(pos_idxs), 250), replace=False))
            neg_idxs = list(rng.choice([idx for idx, label in enumerate(self.los_3day_labels) if label == 0], size=len(pos_idxs), replace=False))

            self.data = [self.data[idx] for idx in pos_idxs + neg_idxs]

            with open(f"{mimiciv_path}eval_tasks/{split}_icu_los_sample_final_ids.json", "w") as f:
                json.dump(list(self.data), f)

        self.tokenizer = tokenizer
        self.split = split
        self.use_los = use_los

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        if self.use_los:
            full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ICU24_{self.split}_los/" + sample_path, "r"))
        else:
            full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ICU24_{self.split}/" + sample_path, "r"))
        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]

        stay_id = sample["stay_id"]
        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=False)

        los_3day_label = sample["los_3day_label"]

        label = {'los_3day': los_3day_label}

        return {
            'label': label,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': False,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }

def extract_ICU_mortality_data(split="test"):
    # RESTRICT_TO_N_HOURS = None
    # SUMMARY_LEVEL = 1
    RESTRICT_TO_N_HOURS = 24
    SUMMARY_LEVEL = 1

    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_ICU_MORT_{split}"
    else:
        patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_ICU_MORT_{split}"
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    val_patient_paths = json.load(open(f"{mimiciv_path}{split}_paths.json", "r"))

    ICU24_timepoint_paths = []
    ICU24_mort_labels = []

    # get already processed stays from patient_folder
    processed_stays = [f[5:-5] for f in os.listdir(patient_folder)]

    for patient_path in tqdm(val_patient_paths):
        stay_id = patient_path.split('/')[-1][5:-5]
        if stay_id in processed_stays:
            continue
        complete_patient_stay = stay_to_ids_dict[stay_id]
        if 'icu_stay_ids' not in complete_patient_stay or len(complete_patient_stay['icu_stay_ids']) == 0:  # skip samples without ICU stay
            continue

        samples = {}
        full_patient_visit, _ = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # extract every 10th timepoints for all ICU stays of the patient
        target_timepoints = []
        for idx, stay in enumerate(full_patient_visit.patient_icu):
            start_time = stay.icustays.iloc[0]['intime'] + pd.Timedelta('1h')  # start at 1h after ICU admission
            end_time = stay.icustays.iloc[0]['outtime']
            death_time = full_patient_visit.patient_adm.admissions.deathtime.values[0]
            if not pd.isnull(death_time):
                end_time = min(end_time, death_time)  # stop at death time

            for target_time in pd.date_range(start_time, end_time, freq='10H'):
                target_timepoint_idx = (target_time - full_patient_visit.patient_adm.admittime)
                target_timepoint_idx = target_timepoint_idx.total_seconds() / 3600 + 1
                # if ed_stay need to add ed_stay_length to hour_idx
                if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] != 0:
                    target_timepoint_idx += complete_patient_stay['ed_stay_hours'] + 2  # +2 for transition idxs
                target_timepoints.append(target_timepoint_idx)

        # generate samples for all timepoints
        for current_timepoint_idx in target_timepoints:
            current_hour_idx = int(current_timepoint_idx) - 1
            ed_stay = full_patient_visit.patient_ed is not None and current_hour_idx <= complete_patient_stay['ed_stay_hours'] + 1
            if ed_stay:
                print(f"ED stay should be over at this point {stay_id}, skipping sample")
                continue

            if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] != 0 and current_hour_idx == complete_patient_stay[
                'ed_stay_hours'] + 1:
                if full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0] == "ADMITTED" or (complete_patient_stay[
                                                                                                       'hosp_stay_hours'] > 0 and full_patient_visit.patient_adm.dischtime > full_patient_visit.patient_ed.outtime):
                    raise ValueError(f"Entered ICU should not happen for 24h ICU stays {stay_id}")
                else:
                    just_admitted = False
            else:
                just_admitted = False

            prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay,
                                                                         restrict_to_N_hours=RESTRICT_TO_N_HOURS)  # patient till current hour

            icu_outtime = None
            for idx, stay in enumerate(full_patient_visit.patient_icu):
                if stay.icustays.iloc[0]['intime'] <= hour_time <= stay.icustays.iloc[0]['outtime']:  # current stay -> save outtime
                    icu_outtime = stay.icustays.iloc[0]['outtime']
                    break

            desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=(
                                                                                                                                                     not ed_stay) or just_admitted,
                                           add_los=False)  # diag_avail describes if ED diagnosis should be included as ED stay is over

            change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx,
                                                                                                              complete_patient_stay, stay_id,
                                                                                                              add_los=False)

            # mortality within 24h label
            final_hosp_disposition = "DIED" if full_patient_visit.patient_adm.admissions.discharge_location.values[0] == "DIED" else "DISCHARGED"
            if final_hosp_disposition == "DIED":
                # if died need to check if died within current ICU stay or not
                time_of_death = full_patient_visit.patient_adm.admissions.deathtime.values[0]
                if np.isnan(time_of_death):
                    continue
                if icu_outtime is not None and time_of_death <= icu_outtime and time_of_death <= hour_time + pd.Timedelta(hours=24):
                    mort_label = 1
                else:
                    mort_label = 0
            else:
                mort_label = 0

            if change_log is not None:
                samples[int(current_timepoint_idx)] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx,
                                                       'mort_label': mort_label}
                ICU24_timepoint_paths.append(f"stay_{stay_id}.json_{int(current_timepoint_idx)}")
                ICU24_mort_labels.append(mort_label)

        if len(samples) > 0:
            with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                json.dump(samples, f)

    with open(f"{mimiciv_path}eval_tasks/{split}_ICU_mort_timepoint_paths.json", "w") as f:
        json.dump(ICU24_timepoint_paths, f)

    with open(f"{mimiciv_path}eval_tasks/{split}_ICU_mort_labels.json", "w") as f:
        json.dump(ICU24_mort_labels, f)

class ICUMortalityDataset(Dataset):
    def __init__(self, tokenizer, split='val', eval_future_mort=False):

        with open(f"{mimiciv_path}eval_tasks/{split}_ICU_mort_timepoint_paths.json", "r") as f:
            self.data = json.load(f)

        # load labels
        with open(f"{mimiciv_path}eval_tasks/{split}_ICU_mort_labels.json", "r") as f:
            self.mort_label = json.load(f)

        print(f"Loaded {len(self.data)} samples for {split} split")

        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_icu_mort_sample_final_ids.json"):
            with open(f"{mimiciv_path}eval_tasks/{split}_icu_mort_sample_final_ids.json", "r") as f:
                self.data = json.load(f)

            if not eval_future_mort and split == 'val':
                # only take 100 pos and 100 neg
                self.data = self.data[:100] + self.data[-100:]

        else:
            rng = np.random.RandomState(42)
            pos_idxs = [idx for idx, label in enumerate(self.mort_label) if label == 1]
            pos_idxs = list(rng.choice(pos_idxs, size=min(len(pos_idxs), 250), replace=False))
            neg_idxs = list(rng.choice([idx for idx, label in enumerate(self.mort_label) if label == 0], size=len(pos_idxs), replace=False))
            self.data = [self.data[idx] for idx in pos_idxs + neg_idxs]
            print(f"Selected {len(self.data)} samples for {split} split")

            with open(f"{mimiciv_path}eval_tasks/{split}_icu_mort_sample_final_ids.json", "w") as f:
                json.dump(list(self.data), f)

        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ICU_MORT_{self.split}/" + sample_path, "r"))

        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]

        stay_id = sample["stay_id"]
        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=False)

        mort_label = sample["mort_label"]

        label = {'mort_label': mort_label}

        return {
            'label': label,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': False,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }


class ICUTimeseriesDataset(Dataset):
    def __init__(self, tokenizer, split='val', icd=False, value_cols=["vitalsigns", "medications"], time_tolerance=1):

        with open(f"{mimiciv_path}eval_tasks/{split}_ICU_mort_timepoint_paths.json", "r") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples for {split} split")

        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_icu_ts_sample_final_ids.json"):
            with open(f"{mimiciv_path}eval_tasks/{split}_icu_ts_sample_final_ids.json", "r") as f:
                self.data = json.load(f)

        else:
            rng = np.random.RandomState(42)
            self.data = rng.choice(self.data, size=500, replace=False)

            with open(f"{mimiciv_path}eval_tasks/{split}_icu_ts_sample_final_ids.json", "w") as f:
                json.dump(list(self.data), f)

        self.tokenizer = tokenizer
        self.split = split
        self.icd = icd
        self.value_cols = value_cols  # column to predict, e.g. ('icu', 'icustay_id', 'chartevents', 'vitalsigns', 'temperature')
        self.time_tolerance = time_tolerance

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_ICU_MORT_{self.split}/" + sample_path, "r"))
        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]

        stay_id = sample["stay_id"]

        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=False)

        # check if current hour is in an ICU stay
        icu_outtime = None
        stay_idx = None
        # if full_patient_visit.patient_icu is not None:
        for idx, stay in enumerate(full_patient_visit.patient_icu):
            if stay.icustays.iloc[0]['intime'] <= prior_time <= stay.icustays.iloc[0]['outtime']:  # has to be True for one of stays for ICU tasks
                in_icu = True
                icu_outtime = stay.icustays.iloc[0]['outtime']
                stay_idx = idx
                break

        assert in_icu == True, f"Hour {hour_idx} not in ICU stay for stay {stay_id} in ICU24hDataset"

        end_hour_idx = hour_idx + 24  # tolerance hours extra to compute time-tolerant metrics at early and late timepoints as well
        relevant_time_window, end_time = restrict_patient_until_hour(full_patient_visit, end_hour_idx, ed_stay=False, restrict_to_N_hours=24)
        relevant_time_window_time_tolerant, end_time_ = restrict_patient_until_hour(full_patient_visit, end_hour_idx + self.time_tolerance,
                                                                                   ed_stay=False, restrict_to_N_hours=24 + self.time_tolerance)

        label = {}
        label_time_tolerant = {}
        if "vitalsigns" in self.value_cols:
            label["vitalsigns"] = relevant_time_window.patient_icu[stay_idx].chartevents['RoutineVitalSigns']
            label_time_tolerant["vitalsigns"] = relevant_time_window_time_tolerant.patient_icu[stay_idx].chartevents['RoutineVitalSigns']

        if "medications" in self.value_cols:
            label["medications"] = relevant_time_window.patient_icu[stay_idx].inputevents
            label_time_tolerant["medications"] = relevant_time_window_time_tolerant.patient_icu[stay_idx].inputevents


        return {
            'label': label,
            'label_time_tolerant': label_time_tolerant,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': False,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }

def extract_random_Hosp_data(split="val", use_los=False):
    # RESTRICT_TO_N_HOURS = None
    # SUMMARY_LEVEL = 1
    RESTRICT_TO_N_HOURS = 24
    SUMMARY_LEVEL = 1

    if RESTRICT_TO_N_HOURS is None:
        patient_folder = f"{mimiciv_path}eval_tasks/all_data_summarylvl1_HOSP_{split}"
    else:
        patient_folder = f"{mimiciv_path}eval_tasks/all_data_{RESTRICT_TO_N_HOURS}_hours_HOSP_{split}"
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    val_patient_paths = json.load(open(f"{mimiciv_path}{split}_paths.json", "r"))

    timepoint_paths = []
    for idx, patient_path in tqdm(enumerate(val_patient_paths)):
        stay_id = patient_path.split('/')[-1][5:-5]
        complete_patient_stay = stay_to_ids_dict[stay_id]
        if 'hosp_stay_hours' not in complete_patient_stay or complete_patient_stay['hosp_stay_hours'] == None or complete_patient_stay[
            'hosp_stay_hours'] <= 0:  # skip samples without Hosp stay
            continue

        samples = {}
        full_patient_visit, _ = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        # extract random timepoint in the hospital stay
        target_timepoint_idx = np.random.randint(1, complete_patient_stay['hosp_stay_hours'] + 1)  # random timepoint in the hospital stay
        # if ed_stay need to add ed_stay_length to hour_idx
        if 'ed_stay_hours' in complete_patient_stay and full_patient_visit.patient_ed is not None:
            target_timepoint_idx += complete_patient_stay['ed_stay_hours'] + 2  # +2 for transition idxs

        # generate samples for all timepoints
        current_hour_idx = int(target_timepoint_idx) - 1
        ed_stay = False

        if 'ed_stay_hours' in complete_patient_stay and complete_patient_stay['ed_stay_hours'] != 0 and current_hour_idx == complete_patient_stay[
            'ed_stay_hours'] + 1:
            if full_patient_visit.patient_ed.ed_stays['disposition'].iloc[0] == "ADMITTED" or (complete_patient_stay[
                                                                                                   'hosp_stay_hours'] > 0 and full_patient_visit.patient_adm.dischtime > full_patient_visit.patient_ed.outtime):
                just_admitted = True
            else:
                continue
        else:
            just_admitted = False

        prior_patient_state, hour_time = restrict_patient_until_hour(full_patient_visit, current_hour_idx, ed_stay,
                                                                     restrict_to_N_hours=RESTRICT_TO_N_HOURS)

        desc = get_patient_description(full_patient_visit, prior_patient_state, hour_time=hour_time, summary_level=SUMMARY_LEVEL, diag_avail=(not ed_stay) or just_admitted,
                                       add_los=use_los)  # diag_avail describes if ED diagnosis should be included as ED stay is over

        change_log, next_patient_state, entered_icu, released_from_icu_idx = retrieve_change_log_for_hour(full_patient_visit, current_hour_idx,
                                                                                                          complete_patient_stay, stay_id,
                                                                                                          add_los=use_los)

        if change_log is not None:
            samples[int(target_timepoint_idx)] = {'desc': desc, 'change_log': change_log, 'stay_id': stay_id, 'hour_idx': current_hour_idx}
            timepoint_paths.append(f"stay_{stay_id}.json_{int(target_timepoint_idx)}")

        if len(samples) > 0:
            with open(f"{patient_folder}/stay_{stay_id}.json", "w") as f:
                json.dump(samples, f)

    with open(f"{mimiciv_path}eval_tasks/{split}_HOSP_timepoint_paths.json", "w") as f:
        json.dump(timepoint_paths, f)
class HospTimeseriesDataset(Dataset):
    def __init__(self, tokenizer, split='val', icd=False, value_cols=["medications", "labs"], time_tolerance=1):

        with open(f"{mimiciv_path}eval_tasks/{split}_HOSP_timepoint_paths.json", "r") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples for {split} split")

        with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
            self.stay_to_ids_dict = json.load(f)

        if os.path.exists(f"{mimiciv_path}eval_tasks/{split}_hosp_ts_sample_final_ids.json"):
            with open(f"{mimiciv_path}eval_tasks/{split}_hosp_ts_sample_final_ids.json", "r") as f:
                self.data = json.load(f)

        else:
            rng = np.random.RandomState(42)
            self.data = rng.choice(self.data, size=500, replace=False)

            with open(f"{mimiciv_path}eval_tasks/{split}_hosp_ts_sample_final_ids.json", "w") as f:
                json.dump(list(self.data), f)

        self.tokenizer = tokenizer
        self.split = split
        self.icd = icd
        self.value_cols = value_cols  # column to predict, e.g. ('icu', 'icustay_id', 'chartevents', 'vitalsigns', 'temperature')
        self.time_tolerance = time_tolerance

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        full_sample = json.load(open(mimiciv_path + f"eval_tasks/all_data_24_hours_HOSP_{self.split}/" + sample_path, "r"))
        hour_idx = int(timepoint_idx) - 1  # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
        sample = full_sample[timepoint_idx]

        stay_id = sample["stay_id"]

        complete_patient_stay = self.stay_to_ids_dict[str(stay_id)]

        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)

        prior_patient_state, prior_time = restrict_patient_until_hour(full_patient_visit, hour_idx, ed_stay=False)

        # Hosp dataset, so these are always False
        ed_stay = False

        end_hour_idx = hour_idx + 24  # tolerance hours extra to compute time-tolerant metrics at early and late timepoints as well
        relevant_time_window, end_time = restrict_patient_until_hour(full_patient_visit, end_hour_idx, ed_stay=ed_stay, restrict_to_N_hours=24)
        relevant_time_window_time_tolerant, end_time_ = restrict_patient_until_hour(full_patient_visit, end_hour_idx + self.time_tolerance,
                                                                                   ed_stay=ed_stay, restrict_to_N_hours=24 + self.time_tolerance)

        label = {}
        label_time_tolerant = {}

        if "medications" in self.value_cols:
            label["medications"] = relevant_time_window.patient_adm.prescriptions
            label_time_tolerant["medications"] = relevant_time_window_time_tolerant.patient_adm.prescriptions

        if "labs" in self.value_cols:
            label["labs"] = relevant_time_window.patient_adm.labevents
            label_time_tolerant["labs"] = relevant_time_window_time_tolerant.patient_adm.labevents

        return {
            'label': label,
            'label_time_tolerant': label_time_tolerant,
            'sample': sample,
            'stay_id': stay_id,
            'hour_idx': hour_idx,
            'complete_patient_stay': complete_patient_stay,
            'ed_stay': False,
            'full_patient_visit': full_patient_visit,
            'prior_patient_state': prior_patient_state,
            'prior_time': prior_time
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Load config for training")
    parser.add_argument('--task', type=str)
    parser.add_argument('--split', type=str, options=['val', 'test'])
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    task = args.task
    split = args.split

    # ICU Imminent Mort (24h):
    if task == "ICU_Mort":
        extract_ICU_mortality_data(split=split)
        ds0 = ICUMortalityDataset(None, split=split)

    # ICU-LOS-3day:
    elif task == "ICU_LOS":
        extract_ICU_val_data_24h(split=split, use_los=False)
        ds1 = ICU24hDataset(None, split=split)

    # Hosp ICD Code Prediction @ Discharge:
    elif task == "Discharge_ICD":
        extract_last_Hosp_data(split=split, use_los=True, )
        ds2 = DischargeDataset(None, split=split, use_los=True)

    # ED Admission Prediction:
    elif task == "ED_ADM":
        extract_ED_data(split=split, at_discharge=False, at_admission=True)
        ds3 = EDDataset(None, split=split, at_discharge=False, predict_adm=True)

    # ED ICD Code Prediction @ Discharge:
    elif task == "ED_ICD":
        extract_ED_data(split=split, at_discharge=True, use_los=True, at_admission=False)
        ds4 = EDDataset(None, split=split, use_los=True)

    # ED Timeseries Prediction:
    elif task == "ED_TS":
        extract_ED_data(split=split, at_discharge=False, at_admission=False)
        ds5 = EDTimeseriesDataset(None, split=split)

    # ICU Timeseries Prediction: (can use same data as ICU Mortality)
    elif task == "ICU_TS":
        ds6 = ICUTimeseriesDataset(None, split=split)

    # Hosp Timeseries Prediction:
    elif task == "HOSP_TS":
        extract_random_Hosp_data(split=split)
        ds7 = HospTimeseriesDataset(None, split=split)