import gzip
import json
import random
import traceback
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from mimic_iv_extraction.paths import mimiciv_path
from model_code.naive_baselines_utils import create_mean_majority_changelog, create_event_freq_changelog

from patient_model.retrieve_change_log import extract_changes_from_changelog
from patient_model.retrieve_patient_model import retrieve_patient_model_

with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
    stay_to_ids_dict = json.load(f)
broken_stay_ids = ['12']  # stayed 0 hours
stay_to_ids_dict = {k: v for k, v in stay_to_ids_dict.items() if k not in broken_stay_ids}

''' Record if a specific event happened (column exists) and was predicted or not, independent of the value '''


def add_binary_samples(gt, pred, metrics_infos, key):
    gt_set = set(gt.columns)
    try:
        pred_set = set(pred.columns)

        # Calculate TP, FP, FN using set operations
        tp = len(pred_set & gt_set)  # Intersection
        fp = len(pred_set - gt_set)  # Predicted but not in ground truth
        fn = len(gt_set - pred_set)  # In ground truth but not predicted

        metrics_infos['event'][key]['tp'] += tp
        metrics_infos['event'][key]['fp'] += fp
        metrics_infos['event'][key]['fn'] += fn
    except Exception as e:
        # save input and stacktrace to error file
        with open(f"error.txt", "a") as f:
            f.write(f"Error in add_binary_samples: {traceback.print_exc()}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"pred: {pred}\n")
            f.write(f"key: {key}\n")
            f.write("-----------------------------\n")

        # assume no predictions were made
        metrics_infos['event'][key]['tp'] += 0
        metrics_infos['event'][key]['fp'] += 0
        metrics_infos['event'][key]['fn'] += len(gt_set)

    return metrics_infos


def add_binary_samples_from_row(gt, pred, metrics_infos, key, col_name):
    elems_gt = set(gt[col_name].values) if len(gt) > 0 else set()
    try:
        # get binary samples
        elems_pred = set(pred[col_name].values) if len(pred) > 0 else set()

        tp = len(elems_gt.intersection(elems_pred))
        fp = len(elems_pred - elems_gt)
        fn = len(elems_gt - elems_pred)

        metrics_infos['event'][key]['tp'] += tp
        metrics_infos['event'][key]['fp'] += fp
        metrics_infos['event'][key]['fn'] += fn
    except Exception as e:
        # save input and stacktrace to error file
        with open(f"error.txt", "a") as f:
            f.write(f"Error in add_binary_samples_from_row: {traceback.print_exc()}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"pred: {pred}\n")
            f.write(f"key: {key}\n")
            f.write(f"col_name: {col_name}\n")
            f.write("-----------------------------\n")

        # assume no predictions were made
        metrics_infos['event'][key]['tp'] += 0
        metrics_infos['event'][key]['fp'] += 0
        metrics_infos['event'][key]['fn'] += len(elems_gt)

    return metrics_infos


def add_binary_samples_from_list(gt, pred, metrics_infos, key):
    try:
        # get binary samples
        correct = set(gt).intersection(pred)
        incorrect = set(pred) - set(gt)
        tp = len(correct)
        fp = len(incorrect)
        fn = len(set(gt) - set(pred))
        # we have 18 possible categories
        tn = 18 - tp - fp - fn
        metrics_infos['diag'][key]['tp'] += tp
        metrics_infos['diag'][key]['fp'] += fp
        metrics_infos['diag'][key]['fn'] += fn
        metrics_infos['diag'][key]['correct'] += (tp + tn)
        metrics_infos['diag'][key]['incorrect'] += (fp + fn)
    except Exception as e:
        # save input and stacktrace to error file
        with open(f"error.txt", "a") as f:
            f.write(f"Error in add_binary_samples_from_list: {traceback.print_exc()}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"pred: {pred}\n")
            f.write(f"key: {key}\n")
            f.write("-----------------------------\n")

        # assume no predictions were made
        metrics_infos['diag'][key]['tp'] += 0
        metrics_infos['diag'][key]['fp'] += 0
        metrics_infos['diag'][key]['fn'] += len(gt)
    return metrics_infos


''' Record the gt and predicted values for a specific event for all instances where the event occurred AND was predicted '''

def normalize(all_events, gt_values, pred_values, prefix, percentile_dict, event):
    # normalize numerical values
    for idx, col in enumerate(all_events): # iterate over all events -> values are in the same order -> gt_values[idx] corresponds to pred_values[idx] and col
        if prefix == f'charts_':
            min, max = percentile_dict[f"charts_{event}"][col]["1"], percentile_dict[f"charts_{event}"][col]["99"]
        else:
            min, max = percentile_dict[f"{prefix}{col}"]["1"], percentile_dict[f"{prefix}{col}"]["99"]
        # clip values to min and max
        gt_values[idx] = np.float64(gt_values[idx]).clip(min, max)
        pred_values[idx] = np.float64(pred_values[idx]).clip(min, max)
        # normalize
        gt_values[idx] = (gt_values[idx] - min) / (max - min)
        pred_values[idx] = (pred_values[idx] - min) / (max - min)

def add_value_samples_num(gt, pred, metrics_infos, key, percentile_dict, prefix, event=None):
    try:
        all_events = list(set(gt.columns).union(pred.columns))
        # true_pos_events = list(set(gt.columns).intersection(pred.columns))
        gt_values = [gt[col].values[0] if col in gt.columns else None for col in all_events] if len(all_events) > 0 else []
        # for pred get all existing preds, otherwise use dataset mean
        pred_values = [pred[col].values[0] if col in pred.columns else None for col in all_events] if len(all_events) > 0 else []
        # if either gt or pred is None, we want maximum error. Maximum error can be computed from percentile dict. To get the maximum error we will set gt to minimum value and pred to maximum value in those cases
        for idx, col in enumerate(all_events):
            if gt_values[idx] is not None and pred_values[idx] is not None: # valid sample, continue
                continue
            if prefix == f'charts_':
                min, max = percentile_dict[f"charts_{event}"][col]["1"], percentile_dict[f"charts_{event}"][col]["99"]
            else:
                min, max = percentile_dict[f"{prefix}{col}"]["1"], percentile_dict[f"{prefix}{col}"]["99"]
            gt_values[idx] = min
            pred_values[idx] = max

        normalize(all_events, gt_values, pred_values, prefix, percentile_dict, event=event)

        # pred_values = pred[true_pos_events].values[0] if len(pred) > 0 else []
        metrics_infos['numerical'][key]['gts'].extend(gt_values)
        metrics_infos['numerical'][key]['preds'].extend(pred_values)
    except Exception as e:
        # save input and stacktrace to error file
        with open(f"error.txt", "a") as f:
            f.write(f"Error in add_value_samples_num: {traceback.print_exc()}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"pred: {pred}\n")
            f.write(f"key: {key}\n")
            f.write("-----------------------------\n")
    return metrics_infos


def add_value_samples_cat(gt, pred, metrics_infos, key):
    try:
        all_events = list(set(gt.columns).union(pred.columns))
        # true_pos_events = list(set(gt.columns).intersection(pred.columns))
        gt_values = [gt[col].values[0] if col in gt.columns else "UNKNOWN" for col in all_events] if len(all_events) > 0 else []  # use "UNKNOWN" for missing values - will always be incorrect
        # pred_values = pred[true_pos_events].values[0] if len(pred) > 0 else []
        pred_values = [pred[col].values[0] if col in pred.columns else "UNKNOWN" for col in all_events] if len(all_events) > 0 else [] # use "UNKNOWN" for missing values - will always be incorrect
        assert len(gt_values) == len(pred_values), f"Length of gt and pred values is not equal: {len(gt_values)} != {len(pred_values)}"
        correct = sum(np.array(gt_values) == np.array(pred_values))
        incorrect = len(pred_values) - correct

        metrics_infos['categorical'][key]['correct'] += correct
        metrics_infos['categorical'][key]['incorrect'] += incorrect
    except Exception as e:
        # save input and stacktrace to error file
        with open(f"error.txt", "a") as f:
            f.write(f"Error in add_value_samples_cat: {traceback.print_exc()}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"pred: {pred}\n")
            f.write(f"key: {key}\n")
            f.write("-----------------------------\n")
    return metrics_infos


def add_value_samples_col_name(gt, pred, metrics_infos, key):
    try:
        # get column name where value is 1 -> only one gt and one pred -> either correct or incorrect
        gt_values = set(gt.columns[gt.values[0] == 1]) if len(gt) > 0 else set()
        pred_values = set(pred.columns[pred.values[0] == 1]) if len(pred) > 0 else set()
        correct = gt_values.intersection(pred_values)
        incorrect = pred_values - gt_values
        metrics_infos['categorical'][key]['correct'] += len(correct)
        metrics_infos['categorical'][key]['incorrect'] += len(incorrect)
    except Exception as e:
        # save input and stacktrace to error file
        with open(f"error.txt", "a") as f:
            f.write(f"Error in add_value_samples_col_name: {traceback.print_exc()}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"pred: {pred}\n")
            f.write(f"key: {key}\n")
            f.write("-----------------------------\n")
    return metrics_infos


def calc_classification_metrics(metrics_infos, key, metrics, met_key):
    tp = metrics_infos[key]['tp']
    fp = metrics_infos[key]['fp']
    fn = metrics_infos[key]['fn']
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    if tp == fp == fn == 0:
        precision = recall = f1 = np.nan

    metrics[met_key][key] = {'precision': precision, 'recall': recall, 'f1': f1}
    return metrics


def calc_regression_metrics(metrics_infos, key, metrics, met_key):
    for elem_key, elem_val in metrics_infos.items():
        if key in elem_key:
            gt = np.array(elem_val['gts']).astype(float)
            pred = np.array(elem_val['preds']).astype(float)
            rmse = np.sqrt(np.mean((gt - pred) ** 2)) if len(gt) > 0 else np.nan
            mae = np.mean(np.abs(gt - pred)) if len(gt) > 0 else np.nan
            metrics[met_key][elem_key] = {'rmse': rmse, 'mae': mae}

    return metrics


def calc_accuracy(metrics_infos, key, metrics, met_key):
    for elem_key, elem_val in metrics_infos.items():
        if key in elem_key:
            correct = elem_val['correct']
            incorrect = elem_val['incorrect']
            accuracy = correct / (correct + incorrect) if correct + incorrect > 0 else np.nan
            metrics[met_key][elem_key] = {'accuracy': accuracy}
    return metrics


def collect_ed_vital_info(ed_vital, y_ed_vital, metrics_infos, percentile_dict):
    # which vitals were predicted
    metrics_infos = add_binary_samples(y_ed_vital, ed_vital, metrics_infos, 'ed_vitals_event')

    # rythm and pain are text, rest are numeric
    y_ed_vital_text = y_ed_vital[[col for col in ['rhythm', 'pain'] if col in y_ed_vital.columns]]
    ed_vital_text = ed_vital[[col for col in ['rhythm', 'pain'] if col in ed_vital.columns]]
    y_ed_vital_num = y_ed_vital.drop(columns=['rhythm', 'pain'], errors='ignore')
    ed_vital_num = ed_vital.drop(columns=['rhythm', 'pain'], errors='ignore')

    # convert to numeric
    try:
        y_ed_vital_num = y_ed_vital_num.apply(pd.to_numeric, errors='raise')
        ed_vital_num = ed_vital_num.apply(pd.to_numeric, errors='raise')
    except ValueError as e:
        print(
            f"Error converting to numeric: {e} in {y_ed_vital_num.columns} and {ed_vital_num.columns} for values {y_ed_vital_num.values} and {ed_vital_num.values}")
        y_ed_vital_num = y_ed_vital_num.apply(pd.to_numeric, errors='coerce')
        ed_vital_num = ed_vital_num.apply(pd.to_numeric, errors='coerce')

    metrics_infos = add_value_samples_num(y_ed_vital_num, ed_vital_num, metrics_infos, 'ed_vital_values_num', percentile_dict, prefix='vital_')
    metrics_infos = add_value_samples_cat(y_ed_vital_text, ed_vital_text, metrics_infos, 'ed_vital_values_text')

    return metrics_infos


def collect_labevents_info(labevents_df, y_labevents_df, metrics_infos, percentile_dict):
    ''' contains both numerical and categorical values '''
    # drop all columns ending with _valueuom
    labevents_df = labevents_df.drop(columns=[col for col in labevents_df.columns if col.endswith('_valueuom')], errors='ignore')
    y_labevents_df = y_labevents_df.drop(columns=[col for col in y_labevents_df.columns if col.endswith('_valueuom')], errors='ignore')
    # which labevents were predicted
    metrics_infos = add_binary_samples(y_labevents_df, labevents_df, metrics_infos, 'labevents_event')

    # split into numerical and categorical values - all columns that can be converted to numbers are considered numerical
    y_labevents_numeric = y_labevents_df.apply(pd.to_numeric, errors='coerce')
    labevents_numeric = labevents_df.apply(pd.to_numeric, errors='coerce')

    # Separate numerical and non-numerical values
    y_labevents_num = y_labevents_numeric.dropna(axis=1, how='all')  # Columns where all values are NaN are not numerical
    labevents_num = labevents_numeric.dropna(axis=1, how='all')

    y_labevents_text = y_labevents_df[y_labevents_num.columns.symmetric_difference(y_labevents_df.columns)]
    labevents_text = labevents_df[labevents_num.columns.symmetric_difference(labevents_df.columns)]

    metrics_infos = add_value_samples_num(y_labevents_num, labevents_num, metrics_infos, 'labevents_values_num', percentile_dict, prefix='lab_')
    metrics_infos = add_value_samples_cat(y_labevents_text, labevents_text, metrics_infos, 'labevents_values_text')

    return metrics_infos


def collect_output_events_info(outputevents_df, y_outputevents_df, metrics_infos, percentile_dict):
    ''' contains both numerical and categorical values '''
    # drop all columns ending with _valueuom
    outputevents_df = outputevents_df.drop(columns=[col for col in outputevents_df.columns if col.endswith('_valueuom')], errors='ignore')
    y_outputevents_df = y_outputevents_df.drop(columns=[col for col in y_outputevents_df.columns if col.endswith('_valueuom')], errors='ignore')
    metrics_infos = add_binary_samples(y_outputevents_df, outputevents_df, metrics_infos, 'outputevents_event')

    # force convert to numeric
    y_outputevents_df = y_outputevents_df.apply(pd.to_numeric, errors='coerce')
    outputevents_df = outputevents_df.apply(pd.to_numeric, errors='coerce')

    metrics_infos = add_value_samples_num(y_outputevents_df, outputevents_df, metrics_infos, 'outputevents_values', percentile_dict, prefix='outputs_')

    return metrics_infos


def collect_microbiologyevents_info(microbiologyevents_df, y_microbiologyevents_df, metrics_infos):
    # which tests on which spec were done/predicted
    # combine test_name and spec_type_desc to get unique test
    gt_tests = set([(test, spec) for test, spec in zip(y_microbiologyevents_df['test_name'], y_microbiologyevents_df['spec_type_desc'])]) if len(
        y_microbiologyevents_df) > 0 else set()
    pred_tests = set([(test, spec) for test, spec in zip(microbiologyevents_df['test_name'], microbiologyevents_df['spec_type_desc'])]) if len(
        microbiologyevents_df) > 0 else set()
    tp = len(gt_tests.intersection(pred_tests))
    fp = len(pred_tests - gt_tests)
    fn = len(gt_tests - pred_tests)
    metrics_infos['event']['microbiologyevents_event']['tp'] += tp
    metrics_infos['event']['microbiologyevents_event']['fp'] += fp
    metrics_infos['event']['microbiologyevents_event']['fn'] += fn

    # values (org_name) prediction
    # overlapping tests
    # tp_tests = list(gt_tests.intersection(pred_tests))
    all_tests = gt_tests.union(pred_tests)
    existing_tests_gt = set(zip(y_microbiologyevents_df['test_name'], y_microbiologyevents_df['spec_type_desc'])) if len(y_microbiologyevents_df) > 0 else set()
    missing_tests_gt = all_tests - existing_tests_gt

    existing_tests_pred = set(zip(microbiologyevents_df['test_name'], microbiologyevents_df['spec_type_desc'])) if len(microbiologyevents_df) > 0 else set()
    missing_tests_pred = all_tests - existing_tests_pred

    # Create a DataFrame for the missing tests with "UNKNOWN" values
    missing_df_gt = pd.DataFrame(
        [{'test_name': test[0], 'spec_type_desc': test[1], 'org_name': 'UNKNOWN'} for test in missing_tests_gt]
    )
    missing_df_pred = pd.DataFrame(
        [{'test_name': test[0], 'spec_type_desc': test[1], 'org_name': 'UNKNOWN'} for test in missing_tests_pred]
    )

    # get gt_values and pred_values for overlapping tests in same order
    y_microbiologyevents_df = y_microbiologyevents_df[y_microbiologyevents_df.apply(lambda x: (x['test_name'], x['spec_type_desc']) in all_tests, axis=1)]
    y_microbiologyevents_df = pd.concat([y_microbiologyevents_df, missing_df_gt], ignore_index=True)

    microbiologyevents_df = microbiologyevents_df[microbiologyevents_df.apply(lambda x: (x['test_name'], x['spec_type_desc']) in all_tests, axis=1)]
    microbiologyevents_df = pd.concat([microbiologyevents_df, missing_df_pred], ignore_index=True)
    # sort by test_name and spec_type_desc in order of tp_tests
    if len(y_microbiologyevents_df) > 0:
        y_microbiologyevents_df = y_microbiologyevents_df.sort_values(by=['test_name', 'spec_type_desc'])
    if len(microbiologyevents_df) > 0:
        microbiologyevents_df = microbiologyevents_df.sort_values(by=['test_name', 'spec_type_desc'])
    assert len(y_microbiologyevents_df) == len(
        microbiologyevents_df), f"Length of gt and pred values is not equal: {len(y_microbiologyevents_df)} != {len(microbiologyevents_df)}"
    gt_values = y_microbiologyevents_df['org_name'].values if len(y_microbiologyevents_df) > 0 else np.array([])
    pred_values = microbiologyevents_df['org_name'].values if len(microbiologyevents_df) > 0 else np.array([])

    correct_org = sum(gt_values == pred_values)
    incorrect_org = len(pred_values) - correct_org

    metrics_infos['categorical']['microbiologyevents_org']['correct'] += correct_org
    metrics_infos['categorical']['microbiologyevents_org']['incorrect'] += incorrect_org

    return metrics_infos


def collect_chartevents_info(chartevents_dict, y_chartevents_dict, metrics_infos, percentile_dict):
    ''' contains both numerical and categorical values '''
    for event in y_chartevents_dict.keys():
        # drop all columns ending with _valueuom
        chartevents_df = chartevents_dict[event].drop(columns=[col for col in chartevents_dict[event].columns if col.endswith('_valueuom')],
                                                      errors='ignore') if event in chartevents_dict else pd.DataFrame()
        y_chartevents_df = y_chartevents_dict[event].drop(columns=[col for col in y_chartevents_dict[event].columns if col.endswith('_valueuom')],
                                                          errors='ignore')
        # which labevents were predicted
        metrics_infos = add_binary_samples(y_chartevents_df, chartevents_df, metrics_infos, f'chartevents_event--{event}')

        # split into numerical and categorical values - all columns that can be converted to numbers are considered numerical
        y_chartevents_numeric = y_chartevents_df.apply(pd.to_numeric, errors='coerce')
        chartevents_numeric = chartevents_df.apply(pd.to_numeric, errors='coerce')

        # Separate numerical and non-numerical values
        y_chartevents_num = y_chartevents_numeric.dropna(axis=1, how='all')  # Columns where all values are NaN are not numerical
        chartevents_num = chartevents_numeric.dropna(axis=1, how='all')

        # drop columns that are not in the percentile_dict (not numerical), treat as categorical
        drop_from_num = set()
        for col in set(y_chartevents_num.columns):
            if f"charts_{event}" not in percentile_dict or col not in percentile_dict[f"charts_{event}"]:
                print(f"Column {col} not found in percentile_dict for event {event}, treated as categorical value")
                drop_from_num.add(col)
                continue
        y_chartevents_num = y_chartevents_num.drop(columns=drop_from_num)

        drop_from_num = set()
        for col in set(chartevents_num.columns):
            if f"charts_{event}" not in percentile_dict or col not in percentile_dict[f"charts_{event}"]:
                print(f"Column {col} not found in percentile_dict for event {event}, treated as categorical value")
                drop_from_num.add(col)
                continue
        chartevents_num = chartevents_num.drop(columns=drop_from_num)

        y_chartevents_text = y_chartevents_df[y_chartevents_num.columns.symmetric_difference(y_chartevents_df.columns)]
        chartevents_text = chartevents_df[chartevents_num.columns.symmetric_difference(chartevents_df.columns)]

        metrics_infos = add_value_samples_num(y_chartevents_num, chartevents_num, metrics_infos, f'chartevents_values_num--{event}', percentile_dict, prefix=f'charts_', event=event)
        metrics_infos = add_value_samples_cat(y_chartevents_text, chartevents_text, metrics_infos, f'chartevents_values_text--{event}')

    return metrics_infos


def get_patient_model_metrics(generated_texts, original_texts, stay_ids, hour_idxs, baseline=False, mean_maj=False):
    metrics_infos = {}
    metrics_infos['event'] = defaultdict(lambda: defaultdict(int))
    metrics_infos['numerical'] = defaultdict(lambda: defaultdict(list))
    metrics_infos['categorical'] = defaultdict(lambda: defaultdict(int))
    metrics_infos['los'] = defaultdict(lambda: defaultdict(list))
    metrics_infos['diag'] = defaultdict(lambda: defaultdict(int))
    metrics_infos['disposition'] = defaultdict(lambda: defaultdict(int))

    mean_maj_dict = json.load(open(f"{mimiciv_path}train_mean_maj_dict.json", "r"))
    with open(f"{mimiciv_path}percentile_dict.json", "r") as f:
        percentile_dict = json.load(f)
    if baseline:
        event_freq = json.load(open(f"{mimiciv_path}train_event_frequencies.json", "r"))
        if mean_maj:
            (ed_vital_mean, ed_pyxis_mean, ed_icd_mean, ed_disposition_mean, transfers_df_mean, services_df_mean, labevents_df_mean, microbiologyevents_df_mean, prescriptions_df_mean,
             procedures_icd_df_mean, icd_mean, disposition_mean, inputevents_mean, outputevents_df_mean, procedures_df_mean, chartevents_dict_mean) = create_mean_majority_changelog(
                mean_maj_dict)
        else:
            (ed_vital_event, ed_pyxis_event, ed_icd_event, ed_disposition_event, transfers_df_event, services_df_event, labevents_df_event, microbiologyevents_df_event, prescriptions_df_event,
             procedures_icd_df_event, icd_event, disposition_event, inputevents_event, outputevents_df_event, procedures_df_event, chartevents_dict_event) = create_event_freq_changelog(
                event_freq, mean_maj_dict)

    # extract changes from generated texts
    for idx, (text_pred, text_original) in tqdm(enumerate(zip(generated_texts, original_texts))):
        stay_id = stay_ids[idx]
        complete_patient_stay = stay_to_ids_dict[str(stay_id)]
        full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)
        final_ed_icd = full_patient_visit.patient_ed.icd_categories.split(";") if full_patient_visit.patient_ed is not None else []
        final_hosp_icd = full_patient_visit.patient_adm.icd_categories.split(";") if full_patient_visit.patient_adm is not None else []
        final_ed_disposition = full_patient_visit.patient_ed.ed_stays.disposition.values[0] if full_patient_visit.patient_ed is not None else None

        ed_stay_hours = complete_patient_stay['ed_stay_hours'] if 'ed_stay_hours' in complete_patient_stay else 0
        if ed_stay_hours is None:
            ed_stay_hours = 0
        hosp_stay_hours = complete_patient_stay['hosp_stay_hours'] if 'hosp_stay_hours' in complete_patient_stay else 0
        if hosp_stay_hours is None:
            hosp_stay_hours = 0

        if final_ed_disposition != "ADMITTED":
            if ed_stay_hours > 0 and hosp_stay_hours > 0 and full_patient_visit.patient_adm.dischtime > full_patient_visit.patient_ed.outtime:  # this considers the admissions with "OTHER" disposition
                final_ed_disposition = "ADMITTED"
        if final_ed_disposition is not None and final_ed_disposition != "ADMITTED" and final_ed_disposition != "DIED":
            final_ed_disposition = "DISCHARGED"

        ed_disposition_hour_idx = int(
            (full_patient_visit.patient_ed.ed_stays.outtime - full_patient_visit.patient_ed.ed_stays.intime).dt.total_seconds().values[
                0] / 3600) if full_patient_visit.patient_ed is not None else 0.
        final_hosp_disposition = ("DIED" if full_patient_visit.patient_adm.admissions.discharge_location.values[
                                                0] == "DIED" else "DISCHARGED") if full_patient_visit.patient_adm is not None else None
        hosp_disposition_hour_idx = int(ed_disposition_hour_idx + (
                full_patient_visit.patient_adm.admissions.dischtime - full_patient_visit.patient_adm.admissions.admittime).dt.total_seconds().values[
            0] / 3600) if full_patient_visit.patient_adm is not None else 0.

        if not baseline:
            (ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df,
             procedures_icd_df, icd, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict, disposition_icu, ed_los, hospital_los, icu_los) = extract_changes_from_changelog(
                text_pred, full_patient_visit)
        else: # baseline does not support ICU disposition
            if mean_maj:
                (ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df,
                 procedures_icd_df, icd, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict) = \
                    (ed_vital_mean, ed_pyxis_mean, ed_icd_mean, ed_disposition_mean, transfers_df_mean, services_df_mean, labevents_df_mean,
                     microbiologyevents_df_mean, prescriptions_df_mean, procedures_icd_df_mean, icd_mean, disposition_mean, inputevents_mean,
                     outputevents_df_mean, procedures_df_mean, chartevents_dict_mean)
            else: #event frequency
                (ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df,
                 procedures_icd_df, icd, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict) = \
                    (ed_vital_event, ed_pyxis_event, ed_icd_event, ed_disposition_event, transfers_df_event, services_df_event, labevents_df_event, microbiologyevents_df_event,
                     prescriptions_df_event, procedures_icd_df_event, icd_event, disposition_event, inputevents_event, outputevents_df_event, procedures_df_event, chartevents_dict_event)


        # predict changes from original texts
        (y_ed_vital, y_ed_pyxis, y_ed_icd, y_ed_disposition, y_transfers_df, y_services_df, y_labevents_df, y_microbiologyevents_df,
         y_prescriptions_df, y_procedures_icd_df, y_icd, y_disposition, y_inputevents, y_outputevents_df, y_procedures_df,
         y_chartevents_dict, y_disposition_icu, ed_los, hospital_los, icu_los) = extract_changes_from_changelog(text_original, full_patient_visit)

        '''normalize numerical values'''

        '''calculate change-based metrics'''
        # These metrics do not capture ICU disposition infos - need to evaluate later within actual downstream task
        # continuous value predictions
        metrics_infos = collect_ed_vital_info(ed_vital, y_ed_vital, metrics_infos, percentile_dict)
        metrics_infos = add_binary_samples(y_ed_pyxis, ed_pyxis, metrics_infos, 'ed_pyxis')
        # binary value predictions
        # prescriptions_df
        metrics_infos = add_binary_samples(y_prescriptions_df, prescriptions_df, metrics_infos, 'prescriptions')
        # procedures_icd_df
        metrics_infos = add_binary_samples_from_row(y_procedures_icd_df, procedures_icd_df, metrics_infos, 'procedures_icd', col_name='long_title')
        if not baseline:
            # inputevents
            for stay_name in set(inputevents.keys()).union(set(y_inputevents.keys())):
                if stay_name in y_inputevents and stay_name in inputevents:
                    metrics_infos = add_binary_samples(y_inputevents[stay_name], inputevents[stay_name], metrics_infos, 'inputevents')
                elif stay_name in inputevents:
                    metrics_infos = add_binary_samples(pd.DataFrame(), inputevents[stay_name], metrics_infos, 'inputevents')
                else:
                    metrics_infos = add_binary_samples(y_inputevents[stay_name], pd.DataFrame(), metrics_infos, 'inputevents')

            # procedures_df
            for stay_name in set(procedures_df.keys()).union(set(y_procedures_df.keys())):
                if stay_name in y_procedures_df and stay_name in procedures_df:
                    metrics_infos = add_binary_samples(y_procedures_df[stay_name], procedures_df[stay_name], metrics_infos, 'procedures')
                elif stay_name in procedures_df:
                    metrics_infos = add_binary_samples(pd.DataFrame(), procedures_df[stay_name], metrics_infos, 'procedures')
                else:
                    metrics_infos = add_binary_samples(y_procedures_df[stay_name], pd.DataFrame(), metrics_infos, 'procedures')

        else: #assume same predictions for all stays
            for stay_name in y_inputevents.keys():
                metrics_infos = add_binary_samples(y_inputevents[stay_name], inputevents, metrics_infos, 'inputevents')
            for stay_name in y_procedures_df.keys():
                metrics_infos = add_binary_samples(y_procedures_df[stay_name], procedures_df, metrics_infos, 'procedures')

        # transfers_df
        assert len(y_transfers_df) <= 1, "Only one transfer event per hour is allowed"
        metrics_infos = add_value_samples_col_name(y_transfers_df, transfers_df, metrics_infos, 'transfers')
        # services_df
        assert len(y_services_df) <= 1, "Only one service event per hour is allowed"
        metrics_infos = add_value_samples_col_name(y_services_df, services_df, metrics_infos, 'services')

        # labevents_df
        metrics_infos = collect_labevents_info(labevents_df, y_labevents_df, metrics_infos, percentile_dict)
        # microbiologyevents_df
        metrics_infos = collect_microbiologyevents_info(microbiologyevents_df, y_microbiologyevents_df, metrics_infos)
        # outputevents_df
        if not baseline:
            for stay_name in set(outputevents_df.keys()).union(set(y_outputevents_df.keys())):
                if stay_name in y_outputevents_df and stay_name in outputevents_df:
                    metrics_infos = collect_output_events_info(outputevents_df[stay_name], y_outputevents_df[stay_name], metrics_infos, percentile_dict)
                elif stay_name in outputevents_df:
                    metrics_infos = collect_output_events_info(outputevents_df[stay_name], pd.DataFrame(), metrics_infos, percentile_dict)
                else:
                    metrics_infos = collect_output_events_info(pd.DataFrame(), y_outputevents_df[stay_name], metrics_infos, percentile_dict)

            # chartevents_dict -> mix of continuous and categorical values
            for stay_name in set(chartevents_dict.keys()).union(set(y_chartevents_dict.keys())):
                if stay_name in y_chartevents_dict and stay_name in chartevents_dict:
                    metrics_infos = collect_chartevents_info(chartevents_dict[stay_name], y_chartevents_dict[stay_name], metrics_infos, percentile_dict)
                elif stay_name in chartevents_dict:
                    metrics_infos = collect_chartevents_info(chartevents_dict[stay_name], {}, metrics_infos, percentile_dict)
                else:
                    metrics_infos = collect_chartevents_info({}, y_chartevents_dict[stay_name], metrics_infos, percentile_dict)
        else:
            for stay_name in y_outputevents_df.keys():
                metrics_infos = collect_output_events_info(outputevents_df, y_outputevents_df[stay_name], metrics_infos, percentile_dict)
            for stay_name in y_chartevents_dict.keys():
                metrics_infos = collect_chartevents_info(chartevents_dict, y_chartevents_dict[stay_name], metrics_infos, percentile_dict)


        ''' end-of-stay predictions '''
        if ed_icd is not None:
            metrics_infos = add_binary_samples_from_list(final_ed_icd, ed_icd, metrics_infos, 'ed_icd')
        if ed_disposition is not None:
            metrics_infos['disposition']['ed_disposition']['correct'] += 1 if ed_disposition == final_ed_disposition else 0
            metrics_infos['los']['ed_disposition_hour']['gts'].append(ed_disposition_hour_idx)
            metrics_infos['los']['ed_disposition_hour']['preds'].append(hour_idxs[idx])

        if icd is not None:
            metrics_infos = add_binary_samples_from_list(final_hosp_icd, icd, metrics_infos, 'icd')
        if disposition is not None and disposition != "ADMITTED TO ICU":
            metrics_infos['disposition']['disposition']['correct'] += 1 if disposition == final_hosp_disposition else 0
            metrics_infos['los']['disposition_hour']['gts'].append(hosp_disposition_hour_idx)
            metrics_infos['los']['disposition_hour']['preds'].append(hour_idxs[idx])

    '''calculate metrics'''

    metrics = {}
    metrics['event'] = {}
    metrics['numerical'] = {}
    metrics['categorical'] = {}
    metrics['los'] = {}
    metrics['diag'] = {}
    metrics['disposition'] = {}

    # ed_vital
    metrics = calc_classification_metrics(metrics_infos['event'], 'ed_vitals_event', metrics, met_key='event')
    metrics = calc_regression_metrics(metrics_infos['numerical'], 'ed_vital_values_num', metrics, met_key='numerical')
    metrics = calc_accuracy(metrics_infos['categorical'], 'ed_vital_values_text', metrics, met_key='categorical')

    # ed_pyxis
    metrics = calc_classification_metrics(metrics_infos['event'], 'ed_pyxis', metrics, met_key='event')

    # prescriptions_df
    metrics = calc_classification_metrics(metrics_infos['event'], 'prescriptions', metrics, met_key='event')

    # procedures_icd_df
    metrics = calc_classification_metrics(metrics_infos['event'], 'procedures_icd', metrics, met_key='event')

    # inputevents
    metrics = calc_classification_metrics(metrics_infos['event'], 'inputevents', metrics, met_key='event')

    # procedures_df
    metrics = calc_classification_metrics(metrics_infos['event'], 'procedures', metrics, met_key='event')

    # transfers_df
    metrics = calc_accuracy(metrics_infos['categorical'], 'transfers', metrics, met_key='categorical')

    # services_df
    metrics = calc_accuracy(metrics_infos['categorical'], 'services', metrics, met_key='categorical')

    # labevents_df
    metrics = calc_classification_metrics(metrics_infos['event'], 'labevents_event', metrics, met_key='event')
    metrics = calc_regression_metrics(metrics_infos['numerical'], 'labevents_values_num', metrics, met_key='numerical')
    metrics = calc_accuracy(metrics_infos['categorical'], 'labevents_values_text', metrics, met_key='categorical')

    # microbiologyevents_df
    metrics = calc_classification_metrics(metrics_infos['event'], 'microbiologyevents_event', metrics, met_key='event')
    metrics = calc_accuracy(metrics_infos['categorical'], 'microbiologyevents_values', metrics, met_key='categorical')

    # outputevents_df
    metrics = calc_classification_metrics(metrics_infos['event'], 'outputevents_event', metrics, met_key='event')
    metrics = calc_regression_metrics(metrics_infos['numerical'], 'outputevents_values', metrics, met_key='numerical')

    # chartevents_dict
    for event in metrics_infos['event'].keys():
        if event.startswith('chartevents_event'):
            metrics = calc_classification_metrics(metrics_infos['event'], event, metrics, met_key='event')
    for event in metrics_infos['numerical'].keys():
        if event.startswith('chartevents_values_num'):
            metrics = calc_regression_metrics(metrics_infos['numerical'], event, metrics, met_key='numerical')
    for event in metrics_infos['categorical'].keys():
        if event.startswith('chartevents_values_text'):
            metrics = calc_accuracy(metrics_infos['categorical'], event, metrics, met_key='categorical')

    # icd
    metrics = calc_accuracy(metrics_infos['diag'], 'ed_icd', metrics, met_key='diag')
    metrics = calc_classification_metrics(metrics_infos['diag'], 'ed_icd', metrics, met_key='diag')
    metrics = calc_accuracy(metrics_infos['diag'], 'icd', metrics, met_key='diag')
    metrics = calc_classification_metrics(metrics_infos['diag'], 'icd', metrics, met_key='diag')
    # disposition
    metrics = calc_accuracy(metrics_infos['disposition'], 'ed_disposition', metrics, met_key='categorical')
    metrics = calc_accuracy(metrics_infos['disposition'], 'disposition', metrics, met_key='categorical')

    # disposition hour
    metrics = calc_regression_metrics(metrics_infos['los'], 'ed_disposition_hour', metrics, met_key='los')
    metrics = calc_regression_metrics(metrics_infos['los'], 'disposition_hour', metrics, met_key='los')

    # macro metrics
    metrics['macro'] = {}
    metrics['macro']['event'] = {}
    metrics['macro']['numerical'] = {}
    metrics['macro']['categorical'] = {}

    def safe_nanmean(values):
        return 0 if np.all(np.isnan(values)) else np.nanmean(values)

    metrics['macro']['event'] = {}
    metrics['macro']['event']['precision'] = safe_nanmean(
        [metrics['event'][key]['precision'] for key in metrics['event'].keys()]
    )
    metrics['macro']['event']['recall'] = safe_nanmean(
        [metrics['event'][key]['recall'] for key in metrics['event'].keys()]
    )
    metrics['macro']['event']['f1'] = safe_nanmean(
        [metrics['event'][key]['f1'] for key in metrics['event'].keys()]
    )

    metrics['macro']['numerical'] = {}
    metrics['macro']['numerical']['rmse'] = safe_nanmean(
        [metrics['numerical'][key]['rmse'] for key in metrics['numerical'].keys()]
    )
    metrics['macro']['numerical']['mae'] = safe_nanmean(
        [metrics['numerical'][key]['mae'] for key in metrics['numerical'].keys()]
    )

    metrics['macro']['categorical']['accuracy'] = safe_nanmean(
        [metrics['categorical'][key]['accuracy'] for key in metrics['categorical'].keys()]
    )

    # calculate micro average metrics for event, numerical and categorical
    metrics['micro'] = {}
    metrics['micro']['event'] = {}
    metrics['micro']['numerical'] = {}
    metrics['micro']['categorical'] = {}

    # event
    tps = sum([metrics_infos['event'][k]['tp'] for k in metrics_infos['event'].keys()])
    fps = sum([metrics_infos['event'][k]['fp'] for k in metrics_infos['event'].keys()])
    fns = sum([metrics_infos['event'][k]['fn'] for k in metrics_infos['event'].keys()])
    metrics['micro']['event']['precision'] = tps / (tps + fps) if tps + fps > 0 else 0
    metrics['micro']['event']['recall'] = tps / (tps + fns) if tps + fns > 0 else 0
    metrics['micro']['event']['f1'] = 2 * metrics['micro']['event']['precision'] * metrics['micro']['event']['recall'] / (
            metrics['micro']['event']['precision'] + metrics['micro']['event']['recall']) if metrics['micro']['event']['precision'] + \
                                                                                             metrics['micro']['event']['recall'] > 0 else 0
    # numerical: RMSE
    gt = np.concatenate([metrics_infos['numerical'][k]['gts'] for k in metrics_infos['numerical'].keys()])
    pred = np.concatenate([metrics_infos['numerical'][k]['preds'] for k in metrics_infos['numerical'].keys()])
    metrics['micro']['numerical']['rmse'] = np.sqrt(safe_nanmean((gt - pred) ** 2))
    metrics['micro']['numerical']['mae'] = safe_nanmean(np.abs(gt - pred))

    # categorical: accuracy
    correct = sum([metrics_infos['categorical'][k]['correct'] for k in metrics_infos['categorical'].keys()])
    incorrect = sum([metrics_infos['categorical'][k]['incorrect'] for k in metrics_infos['categorical'].keys()])
    metrics['micro']['categorical']['accuracy'] = correct / (correct + incorrect) if correct + incorrect > 0 else 0

    return metrics