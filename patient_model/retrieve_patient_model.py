import gzip
import json
import logging
import pickle
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from mimic_iv_extraction.paths import mimiciv_path as mimiciv_path


def retrieve_patient_model_(complete_stay_id, complete_patient_stay, hour_idx=None):
    '''
    Retrieve patient model for a given patient and admission id
    :param complete_stay_id: stay identifier, each stay is either an admission (with linked ED and ICU stays) or an ED stay
    :param ed_stay: if True retrieve time-point during ed_stay, else retrieve time-point during admission
    :param hour_idx: retrieve data relative to the nth hour during hospital stay, if None retrieve all data
    :return: patient representation for the given hour and all static data about the patient

    - how to handle ed stays: all prior ed stays are retrieved as patient history, but patient model always represents a hospital stay, not time-point in ed
    - how to handle icu stays: each icu stay happens within a hospital stay, so icu stays are included in the patient history and current time-point might be in icu
    - how to handle multiple hospital stays: each hospital stay is a separate patient model, but patient history includes all prior hospital stays
    - how to handle missing values: include last valid value for each time-series event in current time-point, mark how many hours ago it was
    '''

    assert hour_idx is None, "Error: hour_idx is not None, this is not supported anymore"
    # Load patient data
    ed_stay_hours = complete_patient_stay['ed_stay_hours'] if 'ed_stay_hours' in complete_patient_stay else 0
    only_ed_stay = complete_patient_stay['adm_id'] == None
    get_ed_stay = only_ed_stay or (hour_idx is not None and ed_stay_hours > 0 and hour_idx <= ed_stay_hours)
    patient_id = complete_patient_stay['patient_id']
    patient_folder = mimiciv_path + 'all_admissions/' + f"patient_{patient_id}/"

    meta_info = pd.read_csv(patient_folder + 'meta_info.csv')
    # convert time columns to datetime
    meta_info['start_time'] = pd.to_datetime(meta_info['start_time'])
    meta_info['end_time'] = pd.to_datetime(meta_info['end_time'])

    patient_path = mimiciv_path + 'all_admissions/' + f"patient_{patient_id}/" + f'patient_stay_{complete_stay_id}.pkl.gz'
    try:
        with gzip.open(patient_path, 'rb') as f:
            patient_visit = pickle.load(f)
    except Exception as e:
        logging.warning(f"Error: could not load {patient_path}")
        raise e

    if get_ed_stay:
        ed_id = complete_patient_stay['ed_stay_id']
        current_visit = meta_info[(meta_info['ed_stay_id'] == ed_id) & (meta_info['stay_type'] == 'ed')]
        assert len(current_visit) <= 1, "Error: multiple ed stays with same id"
        assert len(current_visit) > 0, "Error: no ed stay with given id"

    else:  # admission
        adm_id = complete_patient_stay['adm_id']
        current_visit = meta_info[(meta_info['adm_id'] == adm_id) & (meta_info['stay_type'] == 'admission')]
        assert len(current_visit) <= 1, "Error: multiple admissions with same id"
        assert len(current_visit) > 0, "Error: no admission with given id"
        hour_idx = hour_idx - ed_stay_hours if hour_idx is not None else None

    if get_ed_stay:  # current admission is an ED stay
        current_visit_ed = patient_visit.patient_ed
        current_visit_ed.pre_process()
        if hour_idx is not None:
            # restrict ed data based on hour_idx
            hour_time = current_visit_ed.get_data_until_hour(hour_idx)
        else:
            hour_time = current_visit['end_time']
        current_visit_ed.hour_time = hour_time

        patient_visit = Patient_Visit(patient_ed=current_visit_ed)  # ED visit discharged without admission


    else:  # current admission is an admission
        current_visit_adm = patient_visit.patient_adm
        current_visit_adm.pre_process()

        current_visit_icu = []
        # check if any ICU visits are linked to the current admission in the meta_data
        linked_icu_ids = complete_patient_stay['icu_stay_ids']
        for linked_icu_id in linked_icu_ids:
            patient = patient_visit.patient_icu[linked_icu_id]
            patient.pre_process(current_visit_adm)
            current_visit_icu.append(patient)

        linked_ed_id = complete_patient_stay['ed_stay_id']
        if linked_ed_id is not None:
            current_visit_ed = patient_visit.patient_ed
            current_visit_ed.pre_process()
        else:
            current_visit_ed = None

        if hour_idx is not None:
            # restrict data based on hour_idx
            hour_time = current_visit_adm.get_data_until_hour(hour_idx)
            for icu_visit in current_visit_icu:
                icu_visit.get_data_until_hour(hour_idx)
                icu_visit.hour_time = hour_time
            if current_visit_ed is not None:
                current_visit_ed.get_data_until_hour(hour_idx)
        else:
            hour_time = current_visit['end_time']
            _ = current_visit_adm.get_data_until_hour(hour_idx, hour_time=hour_time.iloc[0])
            for icu_visit in current_visit_icu:
                icu_visit.get_data_until_hour(hour_idx, hour_time=icu_visit.icustays['outtime'].iloc[0])
                icu_visit.hour_time = hour_time
            if current_visit_ed is not None:
                current_visit_ed.get_data_until_hour(hour_idx, hour_time=current_visit_ed.outtime)

        if current_visit_ed is not None:
            current_visit_ed.hour_time = hour_time
        current_visit_adm.hour_time = hour_time
        for icu_visit in current_visit_icu:
            icu_visit.hour_time = hour_time

        patient_visit = Patient_Visit(patient_adm=current_visit_adm, patient_icu=current_visit_icu, patient_ed=current_visit_ed)

    return patient_visit, hour_time


def restrict_patient_until_hour(patient_visit, hour_idx, ed_stay=False, restrict_to_N_hours=None):
    patient_visit_curr = deepcopy(patient_visit)
    if ed_stay:
        hour_time = patient_visit_curr.patient_ed.get_data_until_hour(hour_idx, restrict_to_N_hours=restrict_to_N_hours)
        patient_visit_curr.patient_adm = None
        patient_visit_curr.patient_icu = None

    else:
        if patient_visit_curr.patient_ed is not None:
            patient_visit_curr.patient_ed.get_data_until_hour(hour_idx, restrict_to_N_hours=restrict_to_N_hours)
            ed_stay_len = patient_visit_curr.patient_ed.outtime - patient_visit_curr.patient_ed.intime
            # convert ns timedelta to number of hours
            ed_stay_len = int(ed_stay_len / np.timedelta64(1, 'h')) + 2 #+1 to capture begin and end border and +1 for "just admitted" timestamp and
        else:
            ed_stay_len = 0
        hour_time = patient_visit_curr.patient_adm.get_data_until_hour(hour_idx-ed_stay_len, restrict_to_N_hours=restrict_to_N_hours)         # hour_idx-ed_stay_len should be 0 for the start of the admission
        if patient_visit_curr.patient_icu is not None:
            for idx, _ in enumerate(patient_visit_curr.patient_icu):
                patient_visit_curr.patient_icu[idx].get_data_until_hour(hour_idx-ed_stay_len, restrict_to_N_hours=restrict_to_N_hours)
    return patient_visit_curr, hour_time


def drop_values_flexible(patient_yaml_elem, drop_ratio, return_rest=False):
    if return_rest:
        dropped_elements = {} if isinstance(patient_yaml_elem, dict) else []

        if isinstance(patient_yaml_elem, dict):
            num_to_drop = min(int(len(patient_yaml_elem) * drop_ratio), len(patient_yaml_elem) - 1)
            if num_to_drop > 0:
                keys_to_drop = random.sample(list(patient_yaml_elem.keys()), num_to_drop)
                for key in keys_to_drop:
                    dropped_elements[key] = patient_yaml_elem[key]
                    del patient_yaml_elem[key]

        elif isinstance(patient_yaml_elem, list):
            num_to_drop = min(int(len(patient_yaml_elem) * drop_ratio), len(patient_yaml_elem) - 1)
            if num_to_drop > 0:
                indices_to_drop = set(random.sample(range(len(patient_yaml_elem)), num_to_drop))
                dropped_elements = [elem for i, elem in enumerate(patient_yaml_elem) if i in indices_to_drop]
                patient_yaml_elem = [elem for i, elem in enumerate(patient_yaml_elem) if i not in indices_to_drop]

        else:
            raise ValueError("Unexpected type")

        return patient_yaml_elem, dropped_elements

    else:
        if type(patient_yaml_elem) == dict:
            num_to_drop = min(int(len(patient_yaml_elem) * drop_ratio), len(patient_yaml_elem) - 1)
            if num_to_drop > 0:
                keys_to_drop = random.sample(list(patient_yaml_elem.keys()), num_to_drop)
                for key in keys_to_drop:
                    del patient_yaml_elem[key]

        elif type(patient_yaml_elem) == list:
            num_to_drop = min(int(len(patient_yaml_elem) * drop_ratio), len(patient_yaml_elem) - 1)
            if num_to_drop > 0:
                indices_to_drop = set(random.sample(range(len(patient_yaml_elem)), num_to_drop))
                patient_yaml_elem = [elem for i, elem in enumerate(patient_yaml_elem) if i not in indices_to_drop]

        else:
            raise ValueError("Unexpected type")
        return patient_yaml_elem, None


'''
Estimated ratio between length of characters in the original patient yaml and the tokenized version: 3:1
'''
def summerize_patient_lvl_4(patient_yaml, tokenizer, max_num_tokens, changelog=False, return_rest=False):
    yaml_str = yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False)
    tokenized = tokenizer(yaml_str)
    patient_yaml_rest = None
    if len(tokenized['input_ids']) > max_num_tokens:  # drop random parts of data
        drop_ratio = 0.1
        token_to_char_ratio = len(yaml_str) / len(tokenized['input_ids'])
        max_num_chars = max_num_tokens * token_to_char_ratio

        patient_yaml_original = deepcopy(patient_yaml)
        while len(yaml_str) > max_num_chars and drop_ratio <= 1:
            patient_yaml = deepcopy(patient_yaml_original)
            if return_rest:
                patient_yaml_rest = {}
            if 'Emergency Department Stay' in patient_yaml:
                if 'Medication' in patient_yaml['Emergency Department Stay']:
                    new_yaml, dropped_yaml = drop_values_flexible(
                        patient_yaml['Emergency Department Stay']['Medication'], drop_ratio, return_rest=return_rest)
                    patient_yaml['Emergency Department Stay']['Medication'] = new_yaml
                    if return_rest and len(dropped_yaml) > 0:
                        patient_yaml_rest.setdefault('Emergency Department Stay', {}).setdefault('Medication', {}).update(dropped_yaml)

            if 'Hospital Stay' in patient_yaml:
                if 'Lab Results' in patient_yaml['Hospital Stay']:
                    new_yaml, dropped_yaml = drop_values_flexible(patient_yaml['Hospital Stay']['Lab Results'], drop_ratio, return_rest=return_rest)
                    patient_yaml['Hospital Stay']['Lab Results'] = new_yaml
                    if return_rest and len(dropped_yaml) > 0:
                        patient_yaml_rest.setdefault('Hospital Stay', {}).setdefault('Lab Results', {}).update(dropped_yaml)

                if False and 'Prescriptions' in patient_yaml['Hospital Stay']:
                    new_yaml, dropped_yaml = drop_values_flexible(patient_yaml['Hospital Stay']['Prescriptions'], drop_ratio, return_rest=return_rest)
                    patient_yaml['Hospital Stay']['Prescriptions'] = new_yaml
                    if return_rest and len(dropped_yaml) > 0:
                        patient_yaml_rest.setdefault('Hospital Stay', {}).setdefault('Prescriptions', {}).update(dropped_yaml)

                if 'Procedures' in patient_yaml['Hospital Stay'] and not changelog:
                    num_proc = len(patient_yaml['Hospital Stay']['Procedures'])
                    num_to_delete = int(drop_ratio * num_proc)
                    num_to_keep = num_proc - num_to_delete
                    if return_rest and len(dropped_yaml) > 0:
                        patient_yaml_rest['Hospital Stay'] = {'Procedures': dict(
                        list(patient_yaml['Hospital Stay']['Procedures'].items())[num_to_keep:])}
                    patient_yaml['Hospital Stay']['Procedures'] = dict(
                        list(patient_yaml['Hospital Stay']['Procedures'].items())[:num_to_keep]) # procedure events are time sorted, drop old ones

                if 'Microbiology Growth Results' in patient_yaml['Hospital Stay']:
                    new_yaml, dropped_yaml = drop_values_flexible(patient_yaml['Hospital Stay']['Microbiology Growth Results'], drop_ratio, return_rest=return_rest)
                    patient_yaml['Hospital Stay']['Microbiology Growth Results'] = new_yaml
                    if return_rest and len(dropped_yaml) > 0:
                        patient_yaml_rest.setdefault('Hospital Stay', {}).setdefault('Microbiology Growth Results', {}).update(dropped_yaml)

                if 'Radiology Notes' in patient_yaml['Hospital Stay']: #also time-sorted but not used as output for next timepoint, so drop randomly
                    num_reports = len(patient_yaml['Hospital Stay']['Radiology Notes'])
                    num_to_delete = int(drop_ratio * num_reports)
                    num_to_keep = num_reports - num_to_delete
                    if return_rest and num_to_keep > 0:
                        patient_yaml_rest['Hospital Stay'] = {'Radiology Notes': dict(
                            list(patient_yaml['Hospital Stay']['Radiology Notes'].items())[num_to_keep:])}
                    patient_yaml['Hospital Stay']['Radiology Notes'] = dict(
                        list(patient_yaml['Hospital Stay']['Radiology Notes'].items())[:num_to_keep])

                if 'Outpatient Measurements' in patient_yaml['Hospital Stay']:
                    new_yaml, dropped_yaml = drop_values_flexible(patient_yaml['Hospital Stay']['Outpatient Measurements'], drop_ratio, return_rest=return_rest)
                    patient_yaml['Hospital Stay']['Outpatient Measurements'] = new_yaml
                    if return_rest and len(dropped_yaml) > 0:
                        patient_yaml_rest.setdefault('Hospital Stay', {}).setdefault('Outpatient Measurements', {}).update(dropped_yaml)

            if 'ICU Stay' in patient_yaml:
                for stay in patient_yaml['ICU Stay']:
                    if 'Medication' in patient_yaml['ICU Stay'][stay]:
                        new_yaml, dropped_yaml = drop_values_flexible(patient_yaml['ICU Stay'][stay]['Medication'], drop_ratio, return_rest=return_rest)
                        patient_yaml['ICU Stay'][stay]['Medication'] = new_yaml
                        if return_rest and len(dropped_yaml) > 0:
                            patient_yaml_rest.setdefault('ICU Stay', {}).setdefault(stay, {}).setdefault('Medication', {}).update(dropped_yaml)

                    if 'Output' in patient_yaml['ICU Stay'][stay]:
                        new_yaml, dropped_yaml = drop_values_flexible(patient_yaml['ICU Stay'][stay]['Output'], drop_ratio, return_rest=return_rest)
                        patient_yaml['ICU Stay'][stay]['Output'] = new_yaml
                        if return_rest and len(dropped_yaml) > 0:
                            patient_yaml_rest.setdefault('ICU Stay', {}).setdefault(stay, {}).setdefault('Output', {}).update(dropped_yaml)

                    if 'Procedures' in patient_yaml['ICU Stay'][stay]:
                        new_yaml, dropped_yaml = drop_values_flexible(patient_yaml['ICU Stay'][stay]['Procedures'], drop_ratio, return_rest=return_rest)
                        patient_yaml['ICU Stay'][stay]['Procedures'] = new_yaml
                        if return_rest and len(dropped_yaml) > 0:
                            patient_yaml.setdefault('ICU Stay', {}).setdefault(stay, {}).setdefault('Procedures', {}).update(dropped_yaml)

                    if 'Chart Events' in patient_yaml['ICU Stay'][stay]:
                        all_event_keys = list(patient_yaml['ICU Stay'][stay]['Chart Events'].keys())
                        for event in all_event_keys:
                            new_yaml, dropped_yaml = drop_values_flexible(
                                patient_yaml['ICU Stay'][stay]['Chart Events'][event], drop_ratio, return_rest=return_rest)
                            patient_yaml['ICU Stay'][stay]['Chart Events'][event] = new_yaml
                            if return_rest and len(dropped_yaml) > 0:
                                patient_yaml_rest.setdefault('ICU Stay', {}).setdefault(stay, {}).setdefault('Chart Events', {}).setdefault(event, {}).update(dropped_yaml)

            yaml_str = yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False)
            drop_ratio += 0.1

    if return_rest:
        return patient_yaml, patient_yaml_rest
    else:
        return patient_yaml


def summerize_patient_lvl_4_summary_data(patient_yaml, tokenizer, max_num_tokens, changelog=False):

    yaml_str = yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False)
    tokenized = tokenizer(yaml_str)
    if len(tokenized['input_ids']) > max_num_tokens:  # drop random parts of data
        drop_ratio = 0.1
        token_to_char_ratio = len(yaml_str) / len(tokenized['input_ids'])
        max_num_chars = max_num_tokens * token_to_char_ratio
        patient_yaml_original = deepcopy(patient_yaml)
        while len(yaml_str) > max_num_chars and drop_ratio <= 1:
            # reset patient yaml to original state so drop ratio is applied to the original data not to the already sparse data
            patient_yaml = deepcopy(patient_yaml_original)

            if 'Emergency Department Stay' in patient_yaml:
                if 'Medication' in patient_yaml['Emergency Department Stay']:
                    patient_yaml['Emergency Department Stay']['Medication'], _ = drop_values_flexible(
                        patient_yaml['Emergency Department Stay']['Medication'], drop_ratio)

            if 'Hospital Stay' in patient_yaml:
                if 'Lab Results' in patient_yaml['Hospital Stay']:
                    patient_yaml['Hospital Stay']['Lab Results'], _ = drop_values_flexible(patient_yaml['Hospital Stay']['Lab Results'], drop_ratio)

                if 'Prescriptions' in patient_yaml['Hospital Stay']:
                    patient_yaml['Hospital Stay']['Prescriptions'], _ = drop_values_flexible(patient_yaml['Hospital Stay']['Prescriptions'], drop_ratio)

                if 'Procedures' in patient_yaml['Hospital Stay'] and not changelog:
                    num_to_delete = int(drop_ratio * len(patient_yaml['Procedures']))
                    patient_yaml['Hospital Stay']['Procedures'] = dict(
                        list(patient_yaml['Hospital Stay']['Procedures'].items())[num_to_delete:])

                if 'Microbiology Growth Results' in patient_yaml['Hospital Stay']:
                    patient_yaml['Hospital Stay']['Microbiology Growth Results'], _ = drop_values_flexible(patient_yaml['Hospital Stay']['Microbiology Growth Results'], drop_ratio)

                if 'Radiology Notes' in patient_yaml['Hospital Stay']:
                    patient_yaml['Hospital Stay']['Radiology Notes'], _ = drop_values_flexible(patient_yaml['Hospital Stay']['Radiology Notes'], drop_ratio)

                if 'Outpatient Measurements' in patient_yaml['Hospital Stay']:
                    patient_yaml['Hospital Stay']['Outpatient Measurements'], _ = drop_values_flexible(patient_yaml['Hospital Stay']['Outpatient Measurements'], drop_ratio)

            if 'ICU Stay' in patient_yaml:
                for stay in patient_yaml['ICU Stay']:
                    if 'Medication' in patient_yaml['ICU Stay'][stay]:
                        patient_yaml['ICU Stay'][stay]['Medication'], _ = drop_values_flexible(patient_yaml['ICU Stay'][stay]['Medication'], drop_ratio)

                    if 'Output' in patient_yaml['ICU Stay'][stay]:
                        patient_yaml['ICU Stay'][stay]['Output'], _ = drop_values_flexible(patient_yaml['ICU Stay'][stay]['Output'], drop_ratio)

                    if 'Procedures' in patient_yaml['ICU Stay'][stay]:
                        patient_yaml['ICU Stay'][stay]['Procedures'], _ = drop_values_flexible(patient_yaml['ICU Stay'][stay]['Procedures'], drop_ratio)

                    if 'Chart Events' in patient_yaml['ICU Stay'][stay]:
                        all_event_keys = list(patient_yaml['ICU Stay'][stay]['Chart Events'].keys())
                        for event in all_event_keys:
                            patient_yaml['ICU Stay'][stay]['Chart Events'][event], _ = drop_values_flexible(
                                patient_yaml['ICU Stay'][stay]['Chart Events'][event], drop_ratio)

            yaml_str = yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False)
            drop_ratio += 0.1

        if len(yaml_str) > max_num_chars:
            print("Warning: summary still too long, deleting beginning of data")
            if 'Emergency Department Stay' in patient_yaml:
                if 'Vital Measurements' in patient_yaml['Emergency Department Stay']:
                    key = 'Vital Measurements'
                    num_cols = len(patient_yaml['Emergency Department Stay'][key])
                    for col in patient_yaml['Emergency Department Stay'][key]:
                        patient_yaml['Emergency Department Stay'][key][col] = patient_yaml['Emergency Department Stay'][key][col][int(-max_num_chars//num_cols):]
                if 'Medication' in patient_yaml['Emergency Department Stay']:
                    key = 'Medication'
                    num_cols = len(patient_yaml['Emergency Department Stay'][key])
                    for col in patient_yaml['Emergency Department Stay'][key]:
                        patient_yaml['Emergency Department Stay'][key][col] = patient_yaml['Emergency Department Stay'][key][col][
                                                                              int(-max_num_chars // num_cols):]

            if 'Hospital Stay' in patient_yaml:
                assert len(patient_yaml['Hospital Stay'].keys()) == 1, "Error: multiple keys in patient yaml"
                key = list(patient_yaml['Hospital Stay'].keys())[0]
                num_cols = len(patient_yaml['Hospital Stay'][key])
                for col in patient_yaml['Hospital Stay'][key]:
                    patient_yaml['Hospital Stay'][key][col] = patient_yaml['Hospital Stay'][key][col][int(-max_num_chars//num_cols):]

            if 'ICU Stay' in patient_yaml:
                stay = list(patient_yaml['ICU Stay'].keys())[0]
                assert len(patient_yaml['ICU Stay'][stay].keys()) == 1, "Error: multiple keys in patient yaml"
                key = list(patient_yaml['ICU Stay'][stay].keys())[0]
                if key == "Chart Events":
                    key = list(patient_yaml['ICU Stay'][stay]["Chart Events"].keys())[0]
                    num_cols = len(patient_yaml['ICU Stay'][stay]["Chart Events"][key])
                    for col in patient_yaml['ICU Stay'][stay]["Chart Events"][key]:
                        patient_yaml['ICU Stay'][stay]["Chart Events"][key][col] = patient_yaml['ICU Stay'][stay]["Chart Events"][key][col][int(-max_num_chars//num_cols):]
                else:
                    num_cols = len(patient_yaml['ICU Stay'][stay][key])
                    for col in patient_yaml['ICU Stay'][stay][key]:
                        if type(patient_yaml['ICU Stay'][stay][key][col]) == str:
                            patient_yaml['ICU Stay'][stay][key][col] = patient_yaml['ICU Stay'][stay][key][col][int(-max_num_chars//num_cols):]

    return patient_yaml


def calc_budget_per_column(lengths, L_max):
    lengths_sorted = sorted(lengths, reverse=True)
    remaining_columns = lengths_sorted.copy()
    total_processed_length = 0

    while len(remaining_columns) > 0:
        remaining_budget = L_max - total_processed_length
        budget_per_col = remaining_budget / len(remaining_columns)

        # Find columns within the per-column budget
        within_budget = [l for l in remaining_columns if l <= budget_per_col]

        if len(within_budget) == 0:
            # return current budget per column
            return int(budget_per_col)

        # Move columns within budget to processed_columns
        for l in within_budget:
            remaining_columns.remove(l)
            total_processed_length += l

    # Shorten the remaining columns to the final per-column budget
    remaining_budget = L_max - total_processed_length
    budget_per_col = remaining_budget / len(remaining_columns) if len(remaining_columns) > 0 else remaining_budget

    return int(budget_per_col)

def shorten_description(patient_yaml, tokenizer, max_num_tokens_total):
    yaml_str = yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False)
    tokenized = tokenizer(yaml_str)
    # token_to_char_ratio = len(yaml_str) / len(tokenized['input_ids'])
    # max_num_chars = max_num_tokens_total * token_to_char_ratio

    # calculate how many percent of tokens to drop
    keep_ratio = max_num_tokens_total / len(tokenized['input_ids'])
    max_num_chars = (len(yaml_str) * keep_ratio) * 0.9 # 10% buffer for the case that the tokenization ratio is not perfectly accurate

    if len(yaml_str) > max_num_chars:
        print("Warning: summary still too long, deleting beginning of data")
        if 'Emergency Department Stay' in patient_yaml:
            if 'Vital Measurements' in patient_yaml['Emergency Department Stay']:
                key = 'Vital Measurements'
                value = patient_yaml['Emergency Department Stay'][key]
                col_lens = [len(value[col]) for col in value]
                budget_per_col = calc_budget_per_column(col_lens, max_num_chars)
                for col in value:
                    value[col] = value[col][-budget_per_col:]
            if 'Medication' in patient_yaml['Emergency Department Stay']:
                key = 'Medication'
                value = patient_yaml['Emergency Department Stay'][key]
                col_lens = [len(value[col]) for col in value]
                budget_per_col = calc_budget_per_column(col_lens, max_num_chars)
                for col in value:
                    value[col] = value[col][-budget_per_col:]

        if 'Hospital Stay' in patient_yaml:
            assert len(patient_yaml['Hospital Stay'].keys()) == 1, "Error: multiple keys in patient yaml"
            key = list(patient_yaml['Hospital Stay'].keys())[0]
            value = patient_yaml['Hospital Stay'][key]
            col_lens = [len(value[col]) for col in value]
            budget_per_col = calc_budget_per_column(col_lens, max_num_chars)
            for col in value:
                value[col] = value[col][-budget_per_col:]

        if 'ICU Stay' in patient_yaml:
            stay = list(patient_yaml['ICU Stay'].keys())[0]
            assert len(patient_yaml['ICU Stay'][stay].keys()) == 1, "Error: multiple keys in patient yaml"
            key = list(patient_yaml['ICU Stay'][stay].keys())[0]
            if key == "Chart Events":
                key = list(patient_yaml['ICU Stay'][stay]["Chart Events"].keys())[0]
                value = patient_yaml['ICU Stay'][stay]["Chart Events"][key]
                col_lens = [len(value[col]) for col in value]
                budget_per_col = calc_budget_per_column(col_lens, max_num_chars)
                for col in value:
                    value[col] = value[col][-budget_per_col:]

            else:
                value = patient_yaml['ICU Stay'][stay][key]
                col_lens = [len(value[col]) for col in value if type(value[col]) == str]
                budget_per_col = calc_budget_per_column(col_lens, max_num_chars)

                for col in value:
                    if type(value[col]) == str:
                        value[col] = value[col][-budget_per_col:]

    if len(tokenizer(yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False))['input_ids']) > max_num_tokens_total + 1000:
        a = 1

    return patient_yaml


def split_section_into_parts(patient_yaml, tokenizer, max_num_tokens, changelog=False):
    yaml_str = yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False)
    tokenized = tokenizer(yaml_str)
    parts = [patient_yaml]
    patient_yaml = shorten_description(patient_yaml=patient_yaml, tokenizer=tokenizer, max_num_tokens_total=20000)

    if len(tokenized['input_ids']) > max_num_tokens:
        parts = []
        while(tokenized is not None):
            patient_yaml_part, remaining_part = summerize_patient_lvl_4(patient_yaml, tokenizer, max_num_tokens, changelog, return_rest=True)
            parts.append(patient_yaml_part)
            if remaining_part is not None:
                patient_yaml = remaining_part
                yaml_str = yaml.dump(patient_yaml, default_flow_style=False, sort_keys=False)
                tokenized = tokenizer(yaml_str)
            else:
                tokenized = None

    return parts

def get_patient_description(full_patient_visit, patient_visit, hour_time, summary_level=None, diag_avail=False, add_los=False, los_dict=None, los_only=False, force_los=False, ed_stay=False):
    patient_yaml = {}
    if patient_visit.patient_ed is not None:
        ed_outtime = full_patient_visit.patient_ed.outtime if add_los else None
        ed_los = los_dict['ed_los'] if los_dict is not None else None
        patient_yaml['Emergency Department Stay'] = patient_visit.patient_ed.get_description(hour_time, summary_level=summary_level,
                                                                                             diag_avail=diag_avail, out_time=ed_outtime, ed_los=ed_los, los_only=los_only, force_los=force_los)
    if not ed_stay and patient_visit.patient_adm is not None:
        disch_time = full_patient_visit.patient_adm.admissions.dischtime.iloc[0] if add_los else None
        hospital_los = los_dict['hospital_los'] if los_dict is not None else None
        patient_yaml['Hospital Stay'] = patient_visit.patient_adm.get_description(hour_time, summary_level=summary_level, out_time=disch_time, hospital_los=hospital_los, los_only=los_only, force_los=force_los)
    if not ed_stay and patient_visit.patient_icu is not None:
        patient_yaml['ICU Stay'] = {}
        for idx, icu in enumerate(patient_visit.patient_icu):
            if 0 <= idx < len(patient_visit.patient_icu):
                icustays = patient_visit.patient_icu[idx].icustays
                if 'intime' in icustays.columns and not icustays['intime'].empty:
                    icu_intime = icustays['intime'].iloc[0]
                    if hour_time >= icu_intime:
                        icu_outtime = patient_visit.patient_icu[idx].icustays['outtime'].iloc[0] if add_los else None
                        if icu_outtime is not None and icu_outtime > disch_time: # if icu stay ends after hospital discharge, set outtime to discharge time, noisyness in data
                            icu_outtime = disch_time
                        if los_dict is not None and los_dict['icu_los'] is not None and len(los_dict['icu_los']) > 0 and f'Stay {idx}' in los_dict['icu_los']:
                            icu_los = los_dict['icu_los'][f'Stay {idx}']
                        else:
                            # logging.warning(f"Los Dict: {los_dict}")
                            icu_los = None
                        desc = icu.get_description(hour_time, summary_level=summary_level, out_time=icu_outtime, icu_los=icu_los, los_only=los_only)
                        if len(desc) > 0:
                            patient_yaml['ICU Stay'][f'Stay {idx}'] = desc
                else:
                    print(f"WARNING: ICU stay {idx} not found with icu length {len(patient_visit.patient_icu)} and intime length {len(patient_visit.patient_icu[idx].icustays['intime'])}")
            else:
                print(f"WARNING: ICU stay {idx} not found with icu length {len(patient_visit.patient_icu)}")

        if len(patient_yaml['ICU Stay']) == 0:
            patient_yaml.pop('ICU Stay')
    return patient_yaml


def get_patient_changelog(full_patient_visit, next_hour_time, patient_state, entered_icu, released_from_icu_idx, add_los=False, los_dict=None):
    patient_yaml = {}
    if patient_state.patient_ed is not None:
        ed_outtime = full_patient_visit.patient_ed.outtime if add_los else None
        ed_los = los_dict['ed_los'] if los_dict is not None else None
        patient_yaml['Emergency Department Stay'] = patient_state.patient_ed.get_change_description(out_time=ed_outtime, next_hour_time=next_hour_time, ed_los=ed_los)
    if patient_state.patient_adm is not None:
        disch_time = full_patient_visit.patient_adm.admissions.dischtime.iloc[0] if add_los else None
        hospital_los = los_dict['hospital_los'] if los_dict is not None else None
        patient_yaml['Hospital Stay'] = patient_state.patient_adm.get_change_description(entered_icu, out_time=disch_time, next_hour_time=next_hour_time, hospital_los=hospital_los)
    if patient_state.patient_icu is not None:
        patient_yaml['ICU Stay'] = {}
        for idx, icu in enumerate(patient_state.patient_icu):
            icu_intime = full_patient_visit.patient_icu[idx].icustays['intime'].iloc[0]
            if next_hour_time >= icu_intime:
                icu_outtime = full_patient_visit.patient_icu[idx].icustays['outtime'].iloc[0] if add_los else None
                if icu_outtime is not None and icu_outtime > disch_time: # if icu stay ends after hospital discharge, set outtime to discharge time, noisyness in data
                    icu_outtime = disch_time
                icu_los = los_dict['icu_los'][f'Stay {idx}'] if los_dict is not None else None
                patient_yaml['ICU Stay'][f'Stay {idx}'] = icu.get_change_description(released_from_icu=idx == released_from_icu_idx, out_time=icu_outtime, next_hour_time=next_hour_time, icu_los=icu_los)
                if len(patient_yaml['ICU Stay'][f'Stay {idx}']) == 0:
                    del patient_yaml['ICU Stay'][f'Stay {idx}']
        if len(patient_yaml['ICU Stay']) == 0:
            del patient_yaml['ICU Stay']
    return patient_yaml
