# noinspection PyUnresolvedReferences
from unsloth_helpers.patch_unsloth import *

import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict
from itertools import islice

import numpy as np
import pandas as pd
import transformers
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from patient_model.dataset import CustomDataCollatorForSeq2Seq
from model_code.evaluation import calc_classification_metrics, calc_regression_metrics, calc_accuracy
from model_code.extract_summary_embs import load_summ_model
from mimic_iv_extraction.paths import mimiciv_path
from patient_model.downstream_task_datasets import ICU24hDataset, DischargeDataset, AdmissionDataset, EDDataset, ICUMortalityDataset, \
    EDTimeseriesDataset, ICUTimeseriesDataset, HospTimeseriesDataset
from model_code.iterative_inference import simulate_development
# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from patient_model.retrieve_patient_model import restrict_patient_until_hour
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# load model, tokenizer and checkpoint
max_seq_length = 34000  # Choose any! We auto support RoPE Scaling internally! #was 2048
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
use_summarization = True  # Use summarization model. Can be False.
model_name = "unsloth/Qwen2-0.5B-Instruct-bnb-4bit"

all_icd_codes = ['infection', 'neoplasms', 'endocrine', 'blood', 'mental', 'nervous', 'circulatory', 'respiratory', 'digestive',
                 'genitourinary', 'pregnancy', 'skin', 'musculoskeletal', 'congenital', 'perinatal', 'ill_defined', 'injury', 'unknown']

eval_results_folder = "eval_results"

class CustomEvalCollator:
    def __init__(self):
        pass

    def __call__(self, features):
        return features

def calculate_metrics_(gt_labels, pred_labels):
    # set all nans in pred_labels to 0
    pred_labels = np.nan_to_num(pred_labels, nan=0)

    # calculate metrics using sklearn
    accuracy = accuracy_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    f1_score_ = f1_score(gt_labels, pred_labels)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score_}")

    return accuracy, precision, recall, f1_score_

def calculate_metrics(gt_labels, pred_labels):
    metrics = {}
    for task in gt_labels.keys():
        print(f"Metrics for task: {task}")
        task_gt_labels = gt_labels[task]
        task_pred_labels = pred_labels[task]
        if task == "ICD":
            continue
        if task == "ICU_LOS_Mae":
            mae = np.nanmean(np.abs(np.array(task_gt_labels) - np.array(task_pred_labels)))
            metrics[task] = {"mae": mae}
        else:
            accuracy, precision, recall, f1_score_ = calculate_metrics_(task_gt_labels, task_pred_labels)
            try:
                auc = roc_auc_score(task_gt_labels, task_pred_labels)
            except Exception as e:
                print(e)
                auc = None
            metrics[task] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score_, "AUC": auc}

    return metrics

def calculate_multilabel_metrics(gt_labels, pred_labels):
    metrics = {}
    for task in gt_labels.keys():
        print(f"Metrics for task: {task}")
        task_gt_labels = gt_labels[task]
        task_pred_labels = pred_labels[task]
        # labels as list of lists
        metrics = classification_report(task_gt_labels, task_pred_labels, target_names=all_icd_codes, output_dict=True, zero_division='warn')
        # calculate AUC
        try:
            aucs = []
            for i in range(len(all_icd_codes)):
                gt = [gt[i] for gt in task_gt_labels]
                pred = [pred[i] for pred in task_pred_labels]

                # Check if the ground truth has both classes (0 and 1)
                if len(set(gt)) > 1:
                    auc = roc_auc_score(gt, pred)
                else:
                    auc = float('nan')  # Assign NaN if AUC is undefined

                aucs.append(auc)

            metrics["AUC"] = aucs
        except Exception as e:
            print(e)
            metrics["AUC"] = None

    return metrics

def calculate_ts_metrics(gt_labels, pred_labels, time_tolerance=1, prior_times=None):
    metrics = {}
    metrics['event'] = {}
    metrics['numerical'] = {}
    metrics['categorical'] = {}

    metrics_infos = {}
    metrics_infos['event'] = defaultdict(lambda: defaultdict(int))
    metrics_infos['numerical'] = defaultdict(lambda: defaultdict(list))
    metrics_infos['categorical'] = defaultdict(lambda: defaultdict(int))

    with open(f"{mimiciv_path}percentile_dict.json", "r") as f:
        percentile_dict = json.load(f)

    for task in gt_labels.keys():
        print(f"Metrics for task: {task}")
        task_gt_labels = gt_labels[task]
        task_pred_labels = pred_labels[task]

        for idx, (pred, gt) in enumerate(zip(task_pred_labels, task_gt_labels)):
            if pred is None: # like this, code only needs to be able to handle an empty dataframe
                pred = pd.DataFrame(columns=['charttime'])
            else:
                # get last hour of prior values
                # add 'charttime' column to pred, values are start_hour + num hours in time col
                if prior_times is not None:
                    start_time = prior_times[task][idx] + pd.to_timedelta(1, unit='h')
                    pred['charttime'] = start_time + pd.to_timedelta(pred['time'], unit='h')
                    # drop time column
                    pred = pred.drop(columns=['time'], errors='ignore')

            if task == "ed_vitalsigns":
                metrics_infos = collect_ed_vital_info(pred, gt, metrics_infos, percentile_dict, time_tolerance=time_tolerance)

            elif task == "ed_medications":
                gt, gt_time_tolerant = gt
                metrics_infos = add_binary_samples_time_tolerant(gt, gt_time_tolerant, pred, metrics_infos, 'ed_pyxis', time_tolerance)

            elif task == "icu_vitalsigns":
                metrics_infos = collect_icu_vitals_info(pred, gt, metrics_infos, percentile_dict, time_tolerance)

            elif task == "icu_medications":  #also for hosp_medications (same logic)
                gt, gt_time_tolerant = gt
                metrics_infos = add_binary_samples_time_tolerant(gt, gt_time_tolerant, pred, metrics_infos, 'inputevents', time_tolerance, time_col='time')

            elif task == "icu_labs": # also hosp_labs
                metrics_infos = collect_labevents_info(pred, gt, metrics_infos, percentile_dict, time_tolerance)


        if task == "ed_vitalsigns":
            metrics = calc_classification_metrics(metrics_infos['event'], 'ed_vitals_event', metrics, met_key='event')
            metrics = calc_regression_metrics(metrics_infos['numerical'], 'ed_vital_values_num', metrics, met_key='numerical')
            metrics = calc_accuracy(metrics_infos['categorical'], 'ed_vital_values_text', metrics, met_key='categorical')

        elif task == "ed_medications":
            metrics = calc_classification_metrics(metrics_infos['event'], 'ed_pyxis', metrics, met_key='event')

        elif task == "icu_vitalsigns":
            metrics = calc_classification_metrics(metrics_infos['event'], 'icu_vitals_event', metrics, met_key='event')
            metrics = calc_regression_metrics(metrics_infos['numerical'], 'icu_vital_values_num', metrics, met_key='numerical')
            metrics = calc_accuracy(metrics_infos['categorical'], 'icu_vital_values_text', metrics, met_key='categorical')

        elif task == "icu_medications": #also for hosp_medications (same logic)
            metrics = calc_classification_metrics(metrics_infos['event'], 'inputevents', metrics, met_key='event')

        elif task == "icu_labs": # also hosp_labs
            metrics = calc_classification_metrics(metrics_infos['event'], 'labevents_event', metrics, met_key='event')
            metrics = calc_regression_metrics(metrics_infos['numerical'], 'labevents_values_num', metrics, met_key='numerical')
            metrics = calc_accuracy(metrics_infos['categorical'], 'labevents_values_text', metrics, met_key='categorical')

    return metrics

def collect_icu_vitals_info(icu_vital, y_icu_vital, metrics_infos, percentile_dict, time_tolerance=1):
    # which vitals were predicted
    y_icu_vital, y_icu_vital_time_tolerant = y_icu_vital
    y_icu_vital = y_icu_vital.drop(columns=[col for col in y_icu_vital.columns if col.endswith('_valueuom')], errors='ignore')
    y_icu_vital_time_tolerant = y_icu_vital_time_tolerant.drop(columns=[col for col in y_icu_vital_time_tolerant.columns if col.endswith('_valueuom')], errors='ignore')
    icu_vital = icu_vital.drop(columns=[col for col in icu_vital.columns if col.endswith('_valueuom')], errors='ignore')

    metrics_infos = add_binary_samples_time_tolerant(y_icu_vital, y_icu_vital_time_tolerant, icu_vital, metrics_infos, 'icu_vitals_event', time_tolerance)

    # split into numerical and categorical values - all columns that can be converted to numbers are considered numerical
    y_icu_vital_numeric = y_icu_vital.apply(pd.to_numeric, errors='coerce')
    icu_vital_numeric = icu_vital.apply(pd.to_numeric, errors='coerce')

    # Separate numerical and non-numerical values
    y_icu_vital_num = y_icu_vital_numeric.dropna(axis=1, how='all')  # Columns where all values are NaN are not numerical
    icu_vital_num = icu_vital_numeric.dropna(axis=1, how='all')

    y_icu_vital_text = y_icu_vital[y_icu_vital_num.columns.symmetric_difference(y_icu_vital.columns.drop('charttime'))]
    icu_vital_text = icu_vital[icu_vital_num.columns.symmetric_difference(icu_vital.columns.drop('charttime'))]

    # deep copy for normalization
    y_icu_vital_num_norm = y_icu_vital_num.copy()
    icu_vital_num_norm = icu_vital_num.copy()

    # normalize numerical values
    for col in set(y_icu_vital_num.columns).intersection(set(icu_vital_num.columns)):
        if col in percentile_dict[f"charts_RoutineVitalSigns"]:
            min, max = percentile_dict[f"charts_RoutineVitalSigns"][col]["1"], percentile_dict[f"charts_RoutineVitalSigns"][col]["99"]
            # clip values to min and max
            y_icu_vital_num[col] = y_icu_vital_num[col].clip(min, max)
            icu_vital_num[col] = icu_vital_num[col].clip(min, max)
            y_icu_vital_num_norm[col] = y_icu_vital_num_norm[col].clip(min, max)
            icu_vital_num_norm[col] = icu_vital_num_norm[col].clip(min, max)

            # normalize
            y_icu_vital_num_norm[col] = (y_icu_vital_num[col] - min) / (max - min)
            icu_vital_num_norm[col] = (icu_vital_num[col] - min) / (max - min)
        else:
            print(col)

    metrics_infos = add_value_samples_num_time_tolerant(y_icu_vital_num, icu_vital_num, metrics_infos, 'icu_vital_values_num', time_tolerance)
    metrics_infos = add_value_samples_num_time_tolerant(y_icu_vital_num_norm, icu_vital_num_norm, metrics_infos, 'icu_vital_values_num_norm', time_tolerance)
    metrics_infos = add_value_samples_cat_time_tolerant(y_icu_vital_text, icu_vital_text, metrics_infos, 'icu_vital_values_text', time_tolerance)

    return metrics_infos

def collect_ed_vital_info(ed_vital, y_ed_vital, metrics_infos, percentile_dict, time_tolerance=1):
    # which vitals were predicted
    y_ed_vital, y_ed_vital_time_tolerant = y_ed_vital
    metrics_infos = add_binary_samples_time_tolerant(y_ed_vital, y_ed_vital_time_tolerant, ed_vital, metrics_infos, 'ed_vitals_event', time_tolerance)

    if ed_vital is not None and y_ed_vital is not None: # if no predictions or no gt, skip, as there is no data to compare and here we only compare predicted data
        # rythm and pain are text, rest are numeric
        y_ed_vital_text = y_ed_vital_time_tolerant[[col for col in ['charttime', 'rhythm', 'pain'] if col in y_ed_vital_time_tolerant.columns]]
        ed_vital_text = ed_vital[[col for col in ['charttime', 'rhythm', 'pain'] if col in ed_vital.columns]]
        y_ed_vital_num = y_ed_vital_time_tolerant.drop(columns=['rhythm', 'pain'], errors='ignore')
        ed_vital_num = ed_vital.drop(columns=['rhythm', 'pain'], errors='ignore')

        # convert to numeric
        y_ed_vital_num = y_ed_vital_num.apply(pd.to_numeric, errors='coerce')
        ed_vital_num = ed_vital_num.apply(pd.to_numeric, errors='coerce')

        # deep copy for normalization
        y_ed_vital_num_norm = y_ed_vital_num.copy()
        ed_vital_num_norm = ed_vital_num.copy()

        # normalize numerical values
        for col in set(y_ed_vital_num.columns).intersection(set(ed_vital_num.columns)):
            if col == 'charttime' or col == 'subject_id' or col == 'stay_id':
                continue

            min, max = percentile_dict[f"vital_{col}"]["1"], percentile_dict[f"vital_{col}"]["99"]

            # clip values to min and max
            y_ed_vital_num[col] = y_ed_vital_num[col].astype(float).clip(min, max)
            ed_vital_num[col] = ed_vital_num[col].astype(float).clip(min, max)
            y_ed_vital_num_norm[col] = y_ed_vital_num_norm[col].astype(float).clip(min, max)
            ed_vital_num_norm[col] = ed_vital_num_norm[col].astype(float).clip(min, max)

           # normalize
            y_ed_vital_num_norm[col] = (y_ed_vital_num_norm[col] - min) / (max - min)
            ed_vital_num_norm[col] = (ed_vital_num_norm[col] - min) / (max - min)

        metrics_infos = add_value_samples_num_time_tolerant(y_ed_vital_num, ed_vital_num, metrics_infos, 'ed_vital_values_num')
        metrics_infos = add_value_samples_num_time_tolerant(y_ed_vital_num_norm, ed_vital_num_norm, metrics_infos, 'ed_vital_values_num_norm')
        metrics_infos = add_value_samples_cat_time_tolerant(y_ed_vital_text, ed_vital_text, metrics_infos, 'ed_vital_values_text')

    return metrics_infos

def collect_labevents_info(labevents_df, y_labevents_df, metrics_infos, percentile_dict, time_tolerance=1):
    ''' contains both numerical and categorical values '''
    # drop all columns ending with _valueuom
    y_labevents, y_labevents_time_tolerant = y_labevents_df
    labevents_df = labevents_df.drop(columns=[col for col in labevents_df.columns if col.endswith('_valueuom') or col.endswith('_flag') or col.endswith('_priority') or col.endswith('_ref_range_upper') or col.endswith('_ref_range_lower') or col.endswith('_comments')], errors='ignore') if labevents_df is not None else None
    y_labevents = y_labevents.drop(columns=[col for col in y_labevents.columns if col.endswith('_valueuom') or col.endswith('_flag') or col.endswith('_priority') or col.endswith('_ref_range_upper') or col.endswith('_ref_range_lower') or col.endswith('_comments')], errors='ignore') if y_labevents is not None else None
    # also drop all ending with _flag _priority _ref_range_upper _ref_range_lower _comments
    y_labevents_time_tolerant = y_labevents_time_tolerant.drop(columns=[col for col in y_labevents_time_tolerant.columns if col.endswith('_valueuom') or col.endswith('_flag') or col.endswith('_priority') or col.endswith('_ref_range_upper') or col.endswith('_ref_range_lower') or col.endswith('_comments')], errors='ignore') if y_labevents_time_tolerant is not None else None
    # which labevents were predicted
    metrics_infos = add_binary_samples_time_tolerant(gt=y_labevents, gt_time_tolerant=y_labevents_time_tolerant, pred=labevents_df, metrics_infos=metrics_infos, key='labevents_event', time_tolerance=time_tolerance)

    if labevents_df is not None and y_labevents is not None:
        # split into numerical and categorical values - all columns that can be converted to numbers are considered numerical
        y_labevents_numeric = y_labevents.apply(pd.to_numeric, errors='coerce')
        labevents_numeric = labevents_df.apply(pd.to_numeric, errors='coerce')

        # Separate numerical and non-numerical values
        y_labevents_num = y_labevents_numeric.dropna(axis=1, how='all')  # Columns where all values are NaN are not numerical
        labevents_num = labevents_numeric.dropna(axis=1, how='all')

        y_labevents_text = y_labevents[y_labevents_num.columns.symmetric_difference(y_labevents.columns.drop('charttime'))]
        labevents_text = labevents_df[labevents_num.columns.symmetric_difference(labevents_df.columns.drop('charttime'))]

        # deep copy for normalization
        y_labevents_num_norm = y_labevents_num.copy()
        labevents_num_norm = labevents_num.copy()

        # normalize numerical values
        for col in set(y_labevents_num.columns).intersection(set(labevents_num.columns)):
            if f"lab_{col}" in percentile_dict:
                min, max = percentile_dict[f"lab_{col}"]["1"], percentile_dict[f"lab_{col}"]["99"]
                # clip values to min and max
                y_labevents_num[col] = y_labevents_num[col].clip(min, max)
                labevents_num[col] = labevents_num[col].clip(min, max)
                y_labevents_num_norm[col] = y_labevents_num_norm[col].clip(min, max)
                labevents_num_norm[col] = labevents_num_norm[col].clip(min, max)

                # normalize
                y_labevents_num_norm[col] = (y_labevents_num[col] - min) / (max - min)
                labevents_num_norm[col] = (labevents_num[col] - min) / (max - min)
            else:
                print(col)

        metrics_infos = add_value_samples_num_time_tolerant(y_labevents_num, labevents_num, metrics_infos, 'labevents_values_num', time_tolerance)
        metrics_infos = add_value_samples_num_time_tolerant(y_labevents_num_norm, labevents_num_norm, metrics_infos, 'labevents_values_num_norm', time_tolerance)
        metrics_infos = add_value_samples_cat_time_tolerant(y_labevents_text, labevents_text, metrics_infos, 'labevents_values_text', time_tolerance)

    return metrics_infos

def extract_events(df, time_col = "charttime"):
    """Extract a list of (time, event_type) for non-null events in df."""
    events = []
    if df is None or len(df) == 0:
        return events
    for col in df.columns:
        if col == time_col or col == 'subject_id' or col == 'stay_id' or '_valueuom' in col or '_rate' in col or '_dose' in col:
            continue
        # For each row, if the event column is truthy, add an event
        for time in df.loc[df[col].notnull(), time_col]:
            events.append((time, col))
    return events

def add_binary_samples_time_tolerant(gt, gt_time_tolerant, pred, metrics_infos, key, time_tolerance=1, time_col='charttime'):
    tolerance = pd.Timedelta(hours=time_tolerance)

    # Extract events as (time, event) pairs
    gt_events = extract_events(gt_time_tolerant, time_col)
    pred_events = extract_events(pred, time_col)

    # Convert ground truth events into a dict keyed by event_type for faster lookup
    gt_dict = {}
    for time, event in gt_events:
        gt_dict.setdefault(event, []).append(time)
    # Sort times for each event type
    for times in gt_dict.values():
        times.sort()

    tp = 0
    unmatched_pred = []

    # For each predicted event, check for a matching gt event within tolerance
    for p_time, p_event in pred_events:
        matched = False
        if p_event in gt_dict:
            # Binary search can be used, but for simplicity, iterate over possible times
            for g_time in gt_dict[p_event]:
                if abs(g_time - p_time) <= tolerance:
                    matched = True
                    break
        if matched:
            tp += 1
        else:
            unmatched_pred.append((p_time, p_event))

    # All predicted events not matched are false positives
    fp = len(unmatched_pred)

    # For false negatives, check which gt events did not get matched by any pred
    # For simplicity, we reuse matching logic for FN:
    gt_events = extract_events(gt, time_col) # here don't use gt_time_tolerant because we don't expect a match for extra events
    fn = 0
    for g_time, g_event in gt_events:
        matched = False
        for p_time, p_event in pred_events:
            if p_event == g_event and abs(g_time - p_time) <= tolerance:
                matched = True
                break
        if not matched:
            fn += 1

    metrics_infos['event'][key]['tp'] += tp
    metrics_infos['event'][key]['fp'] += fp
    metrics_infos['event'][key]['fn'] += fn

    return metrics_infos

def add_value_samples_num_time_tolerant(gt, pred, metrics_infos, key, time_tolerance=1):
    tolerance = pd.Timedelta(hours=time_tolerance)

    # Determine common numerical event columns, excluding 'charttime'
    common_events = list(set(gt.columns).intersection(pred.columns)) if pred is not None else []
    if len(pred) == 0:
        common_events = []
    if 'charttime' in common_events:
        common_events.remove('charttime')

    if 'subject_id' in common_events:
        common_events.remove('subject_id')

    if 'stay_id' in common_events:
        common_events.remove('stay_id')

    # Initialize storage for merged keys if not present
    if 'numerical' not in metrics_infos:
        metrics_infos['numerical'] = {}

    if f"{key}_all" not in metrics_infos['numerical']:
        metrics_infos['numerical'][f"{key}_all"] = {'gts': [], 'preds': []}

    for event in common_events:
        merged_key = f"{key}_{event}"
        if merged_key not in metrics_infos['numerical']:
            metrics_infos['numerical'][merged_key] = {'gts': [], 'preds': []}

    # Ensure time columns are sorted and in datetime format
    if pred is not None and len(pred) > 0:
        pred = pred.sort_values('charttime').reset_index(drop=True)
        pred['charttime'] = pd.to_datetime(pred['charttime'])

    if gt is not None and len(gt) > 0:
        gt = gt.sort_values('charttime').reset_index(drop=True)
        gt['charttime'] = pd.to_datetime(gt['charttime'])

    for event in common_events:
        merged_key = f"{key}_{event}"
        gt_event = gt[['charttime', event]].dropna(subset=[event]).copy()
        pred_event = pred[['charttime', event]].dropna(subset=[event]).copy()

        for _, pred_row in pred_event.iterrows():
            time_lower = pred_row['charttime'] - tolerance
            time_upper = pred_row['charttime'] + tolerance

            # Find GT candidates within the time window
            candidates = gt_event[
                (gt_event['charttime'] >= time_lower) &
                (gt_event['charttime'] <= time_upper)
                ]

            if candidates.empty:
                continue

            candidates = candidates.copy()
            candidates['abs_diff'] = (candidates[event] - pred_row[event]).abs()

            best_match = candidates.loc[candidates['abs_diff'].idxmin()]

            metrics_infos['numerical'][merged_key]['gts'].append(best_match[event])
            metrics_infos['numerical'][merged_key]['preds'].append(pred_row[event])
            metrics_infos['numerical'][f"{key}_all"]['gts'].append(best_match[event])
            metrics_infos['numerical'][f"{key}_all"]['preds'].append(pred_row[event])

    return metrics_infos

def add_value_samples_cat_time_tolerant(gt, pred, metrics_infos, key, time_tolerance=1):
    tolerance = pd.Timedelta(hours=time_tolerance)

    # Identify common categorical event columns excluding 'charttime'
    common_events = list(set(gt.columns).intersection(pred.columns)) if pred is not None else []
    if len(pred) == 0 or len(gt) == 0:
        common_events = []
    if 'charttime' in common_events:
        common_events.remove('charttime')

    # Initialize storage for merged keys if not present
    if 'categorical' not in metrics_infos:
        metrics_infos['categorical'] = {}
    if f"{key}_all" not in metrics_infos['categorical']:
        metrics_infos['categorical'][f"{key}_all"] = {'correct': 0, 'incorrect': 0}

    for event in common_events:
        merged_key = f"{key}_{event}"
        if merged_key not in metrics_infos['categorical']:
            metrics_infos['categorical'][merged_key] = {'correct': 0, 'incorrect': 0}

    for event in common_events:
        merged_key = f"{key}_{event}"

        # Extract non-null rows for the event
        gt_event = gt[['charttime', event]].dropna(subset=[event]).copy()
        pred_event = pred[['charttime', event]].dropna(subset=[event]).copy()

        # If no predictions, skip
        if pred_event.empty or gt_event.empty:
            continue

        # For each prediction, check for a matching ground truth
        for _, pred_row in pred_event.iterrows():
            time_lower = pred_row['charttime'] - tolerance
            time_upper = pred_row['charttime'] + tolerance

            # Find GT events within the time window
            candidates = gt_event[
                (gt_event['charttime'] >= time_lower) &
                (gt_event['charttime'] <= time_upper)
            ]

            if candidates.empty:
                continue

            standardized_gt = candidates[event].astype(str).str.lower().values
            if key == 'ed_vital_values_text':
                if event == 'pain':
                    if "unable" in pred_row[event].lower() or 'uta' in pred_row[event].lower() or 'u/a' in pred_row[event].lower():
                        pred_row[event] = 'unable to assess'
                    if "sleep" in pred_row[event].lower(): #sleeping and asleep
                        pred_row[event] = 'sleeping'
                    if '/10' in pred_row[event]:
                        pred_row[event] = pred_row[event].split('/')[0] + ".0"

                    standardized_gt = ['unable to assess' if "unable" in val or 'uta' in val or 'u/a' in val else val for val in standardized_gt]
                    standardized_gt = ['sleeping' if "sleep" in val else val for val in standardized_gt]

                elif event == 'rhythm':
                    if 'avp' in pred_row[event].lower():
                        pred_row[event] = 'av paced'

                    standardized_gt = ['av paced' if 'avp' in val else val for val in standardized_gt]


            # Check if any candidate has the same category value
            if str(pred_row[event]).lower() in standardized_gt:
                metrics_infos['categorical'][merged_key]['correct'] += 1
                metrics_infos['categorical'][f"{key}_all"]['correct'] += 1
            else:
                metrics_infos['categorical'][merged_key]['incorrect'] += 1
                metrics_infos['categorical'][f"{key}_all"]['incorrect'] += 1

    return metrics_infos


'''
only_gt: if True, only return ground truth labels and don't simulate development -> used for calculating metrics
task: "ICU_Mort" or "ICU_LOS"
'''
def eval_ICU(tokenizer, model, split='val', only_gt=False, use_los=True, use_summ=False, summ_model=None, summ_tokenizer=None, summ_collator=None,
             summ_only=False, eval_future_los=False, model_path=""):
    eval_data_collator = CustomEvalCollator()

    eval_dataset = ICU24hDataset(tokenizer, split=split, use_los=use_los, eval_future_los=eval_future_los)
    base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if use_summ:
        batch_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['desc_emb', 'sum_mask'])
    else:
        batch_collator = base_collator

    batch_size = 8
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator)

    if os.path.exists(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_los/current_iter_{model_path.replace('/', '_')}_{split}.txt"):
        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_los/current_iter_{model_path.replace('/', '_')}_{split}.txt", "r") as f:
            current_iter = int(f.read())

        gt_labels = pickle.load(open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_los/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        pred_labels = pickle.load(open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_los/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))

        print("Starting from iteration", current_iter)
        print(f"Loaded gt and pred labels: {gt_labels}, {pred_labels}")

    else:
        current_iter = -1
        gt_labels = defaultdict(list)
        pred_labels = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if batch == [{}]:
            continue
        if i <= current_iter:
            continue

        sample = [b['sample'] for b in batch]
        labels = [b['label'] for b in batch]
        stay_id = [b['stay_id'] for b in batch]
        hour_idx = [b['hour_idx'] for b in batch]
        complete_patient_stay = [b['complete_patient_stay'] for b in batch]
        full_patient_visit = [b['full_patient_visit'] for b in batch]
        prior_patient_state = [b['prior_patient_state'] for b in batch]
        prior_time = [b['prior_time'] for b in batch]

        for j in range(len(batch)):
            gt_labels["ICU_LOS_3day"].append(labels[j]["los_3day"])

        if only_gt:
            continue

        just_admitted = [False]*batch_size
        ed_stay = [False]*batch_size

        batch_output = simulate_development(full_patient_visit=full_patient_visit, sample=sample, stay_id=stay_id, model=model, tokenizer=tokenizer,
                                                                 model_name=model_name, hour_idx=hour_idx, complete_patient_stay=complete_patient_stay,
                                                                 prior_patient_state=prior_patient_state, prior_time=prior_time, icu_only=True,
                                                                 just_admitted=just_admitted, ed_stay=ed_stay, batch_collator=batch_collator, max_steps=73, use_summ=use_summ,
                                                                 summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only,
                                                                 eval_future_los=eval_future_los)

        for j in range(len(batch)):
            _, _, disposition, released_from_icu, icd, new_hour_idx, _ = batch_output[j]
            if eval_future_los:
                if disposition == "DISCHARGED":
                    pred_labels["ICU_LOS_3day"].append(0)
                elif disposition == "STAYED":
                    pred_labels["ICU_LOS_3day"].append(1)
                else:
                    print("Unknown disposition:", disposition)
                    pred_labels["ICU_LOS_3day"].append(0)
            else:
                if released_from_icu:
                    stay_length = new_hour_idx - hour_idx[j]
                    if stay_length > 72:
                        pred_labels["ICU_LOS_3day"].append(1)
                    else:
                        pred_labels["ICU_LOS_3day"].append(0)
                elif disposition == "DIED":
                    pred_labels["ICU_LOS_3day"].append(np.nan)

                else: #reached max steps without being released from ICU
                    pred_labels["ICU_LOS_3day"].append(1)

        # save current gt and pred labels
        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_los/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_los/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)

        # save the current iteration to know where to restart from
        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_los/current_iter_{model_path.replace('/', '_')}_{split}.txt", "w") as f:
            f.write(str(i))

        print(f"Saved current gt and pred labels at iteration {i} of {len(dataloader)}")

    return gt_labels, pred_labels

def eval_ICU_mort(tokenizer, model, split='val', only_gt=False, use_summ=False, summ_model=None, summ_tokenizer=None, summ_collator=None, summ_only=False, eval_future_mort=False, model_path=""):
    eval_data_collator = CustomEvalCollator()

    eval_dataset = ICUMortalityDataset(tokenizer, split=split, eval_future_mort=eval_future_mort)
    base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if use_summ:
        batch_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['desc_emb', 'sum_mask'])
    else:
        batch_collator = base_collator

    batch_size = 8
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator)

    if os.path.exists(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_mort/current_iter_{model_path.replace('/', '_')}_{split}.txt"):
        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_mort/current_iter_{model_path.replace('/', '_')}_{split}.txt", "r") as f:
            current_iter = int(f.read())

        gt_labels = pickle.load(open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_mort/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        pred_labels = pickle.load(open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_mort/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))

    else:
        current_iter = -1
        gt_labels = defaultdict(list)
        pred_labels = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if batch == [{}]:
            continue
        if i <= current_iter:
            continue

        sample = [b['sample'] for b in batch]
        labels = [b['label'] for b in batch]
        stay_id = [b['stay_id'] for b in batch]
        hour_idx = [b['hour_idx'] for b in batch]
        complete_patient_stay = [b['complete_patient_stay'] for b in batch]
        full_patient_visit = [b['full_patient_visit'] for b in batch]
        prior_patient_state = [b['prior_patient_state'] for b in batch]
        prior_time = [b['prior_time'] for b in batch]

        for j in range(len(batch)):
            gt_labels["ICU_Mort"].append(labels[j]["mort_label"])

        if only_gt:
            continue

        just_admitted = [False]*batch_size
        ed_stay = [False]*batch_size


        batch_output = simulate_development(full_patient_visit=full_patient_visit, sample=sample, stay_id=stay_id, model=model, tokenizer=tokenizer,
                                                                model_name=model_name, hour_idx=hour_idx, complete_patient_stay=complete_patient_stay,
                                                                prior_patient_state=prior_patient_state, prior_time=prior_time, icu_only=True,
                                                                just_admitted=just_admitted, ed_stay=ed_stay, batch_collator=batch_collator, use_summ=use_summ,
                                                                summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only, max_steps=24,
                                                                eval_future_mort=eval_future_mort)

        for j in range(len(batch)):
            _, _, disposition, released_from_icu, icd, new_hour_idx, _ = batch_output[j]

            if disposition is not None and disposition == "DIED":
                pred_labels["ICU_Mort"].append(1)
            else:
                pred_labels["ICU_Mort"].append(0)

        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_mort/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_mort/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)

        with open(f"{eval_results_folder}/{'eval_' if split=='val' else ''}icu_mort/current_iter_{model_path.replace('/', '_')}_{split}.txt", "w") as f:
            f.write(str(i))
        print(f"Saved current gt and pred labels at iteration {i} of {len(dataloader)}")

    return gt_labels, pred_labels

def eval_icd(tokenizer, model, split='val', only_gt=False, use_los=False, use_summ=False, summ_model=None, summ_tokenizer=None, summ_collator=None, summ_only=False): #mode can be "ICD" or "Disch"
    eval_data_collator = CustomEvalCollator()
    eval_dataset = DischargeDataset(tokenizer, split=split, use_los=use_los)
    base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if use_summ:
        batch_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['desc_emb', 'sum_mask'])
    else:
        batch_collator = base_collator

    batch_size = 8
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator)

    gt_labels = defaultdict(list)
    pred_labels = defaultdict(list)
    for i, batch in enumerate(dataloader):
        if batch == [{}]:
            continue

        sample = [b['sample'] for b in batch]
        labels = [b['label'] for b in batch]
        stay_id = [b['stay_id'] for b in batch]
        hour_idx = [b['hour_idx'] for b in batch]
        complete_patient_stay = [b['complete_patient_stay'] for b in batch]
        full_patient_visit = [b['full_patient_visit'] for b in batch]
        prior_patient_state = [b['prior_patient_state'] for b in batch]
        prior_time = [b['prior_time'] for b in batch]

        for j in range(len(batch)):
            icd_list = labels[j]["icd"] # this gives a list of categories - need to get the one hot encoding using all_icd_codes
            one_hot_icd = [1 if icd in icd_list else 0 for icd in all_icd_codes]
            gt_labels["ICD"].append(one_hot_icd)

        if only_gt:
            continue

        just_admitted = [False]*len(batch)
        ed_stay = [False]*len(batch)

        batch_outputs = simulate_development(full_patient_visit=full_patient_visit, sample=sample, stay_id=stay_id, model=model, tokenizer=tokenizer,
                                             model_name=model_name, hour_idx=hour_idx, complete_patient_stay=complete_patient_stay,
                                             prior_patient_state=prior_patient_state, prior_time=prior_time, icu_only=False, just_admitted=just_admitted, ed_stay=ed_stay,
                                             batch_collator=batch_collator, use_summ=use_summ, summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only, max_steps=10) #usually is only one step, 10 to avoid endless loop if model makes errors

        for j in range(len(batch)):
            _, _, disposition, released_from_icu, icd, new_hour_idx, _ = batch_outputs[j]

            if icd is not None:
                one_hot_icd = [1 if code in icd else 0 for code in all_icd_codes]
                pred_labels["ICD"].append(one_hot_icd)

            else:
                pred_labels["ICD"].append([0]*len(all_icd_codes))

    return gt_labels, pred_labels

def eval_ed(tokenizer, model, split='val', only_gt=False, at_discharge=False, use_summ=False, use_los=False, summ_model=None, summ_tokenizer=None,
            summ_collator=None, summ_only=False, predict_adm=False, eval_future_ed_disp=False):
    eval_data_collator = CustomEvalCollator()
    eval_dataset = EDDataset(tokenizer, split=split, at_discharge=at_discharge, use_los=use_los, predict_adm=predict_adm, eval_future_ed_disp=eval_future_ed_disp)
    base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    if use_summ:
        batch_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['desc_emb', 'sum_mask'])
    else:
        batch_collator = base_collator

    batch_size = 8
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator)

    gt_labels = defaultdict(list)
    pred_labels = defaultdict(list)
    for i, batch in enumerate(dataloader):
        if batch == [{}]:
            continue

        sample = [b['sample'] for b in batch]
        labels = [b['label'] for b in batch]
        stay_id = [b['stay_id'] for b in batch]
        hour_idx = [b['hour_idx'] for b in batch]
        complete_patient_stay = [b['complete_patient_stay'] for b in batch]
        full_patient_visit = [b['full_patient_visit'] for b in batch]
        prior_patient_state = [b['prior_patient_state'] for b in batch]
        prior_time = [b['prior_time'] for b in batch]

        for j in range(len(batch)):
            gt_labels["ed_adm"].append(labels[j]["ed_adm"])

            icd_list = labels[j]["icd"] # this gives a list of categories - need to get the one hot encoding using all_icd_codes
            one_hot_icd = [1 if icd in icd_list else 0 for icd in all_icd_codes]
            gt_labels["ICD"].append(one_hot_icd)

        if only_gt:
            continue

        just_admitted = [False]*len(batch)
        ed_stay = [True]*len(batch)

        batch_outputs = simulate_development(full_patient_visit=full_patient_visit, sample=sample, stay_id=stay_id, model=model, tokenizer=tokenizer,
                                             model_name=model_name, hour_idx=hour_idx, complete_patient_stay=complete_patient_stay,
                                             prior_patient_state=prior_patient_state, prior_time=prior_time, icu_only=False,
                                             just_admitted=just_admitted, ed_stay=ed_stay, only_ed=True, batch_collator=batch_collator, use_summ=use_summ,
                                             summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only, max_steps=65,
                                             eval_future_ed_disp=eval_future_ed_disp)

        for j in range(len(batch)):
            ed_disposition, ed_icd, disposition, released_from_icu, icd, new_hour_idx, _ = batch_outputs[j]

            # "ICU_Mort" task
            if ed_disposition is not None and ed_disposition == "ADMITTED":
                pred_labels["ed_adm"].append(1)
            else:
                pred_labels["ed_adm"].append(0)

            if ed_icd is not None:
                one_hot_icd = [1 if code in ed_icd else 0 for code in all_icd_codes]
                pred_labels["ICD"].append(one_hot_icd)

            else:
                pred_labels["ICD"].append([0]*len(all_icd_codes))

    return gt_labels, pred_labels

def eval_ed_ts(tokenizer, model, split='val', only_gt=False, value_cols=["vitalsigns", "medications"], time_tolerance=1, use_summ=False, summ_model=None, summ_tokenizer=None, summ_collator=None, summ_only=False):
    eval_data_collator = CustomEvalCollator()
    eval_dataset = EDTimeseriesDataset(None, split=split, value_cols=value_cols, time_tolerance=time_tolerance, at_admission=False)
    base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if use_summ:
        batch_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['desc_emb', 'sum_mask'])
    else:
        batch_collator = base_collator

    batch_size = 8
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator)

    # skip already saved data loading current iteration
    if os.path.exists(f"{eval_results_folder}/ed_ts/current_iter_{model_path.replace('/', '_')}_{split}.txt"):
        with open(f"{eval_results_folder}/ed_ts/current_iter_{model_path.replace('/', '_')}_{split}.txt", "r") as f:
            current_iter = int(f.read())

        gt_labels = pickle.load(open(f"{eval_results_folder}/ed_ts/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        pred_labels = pickle.load(open(f"{eval_results_folder}/ed_ts/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        prior_labels = pickle.load(open(f"{eval_results_folder}/ed_ts/prior_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))

    else:
        current_iter = -1
        gt_labels = defaultdict(list)
        pred_labels = defaultdict(list)
        prior_labels = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if batch == [{}]:
            continue
        if i <= current_iter:
            continue

        sample = [b['sample'] for b in batch]
        labels = [b['label'] for b in batch]
        labels_time_tolerant = [b['label_time_tolerant'] for b in batch]
        stay_id = [b['stay_id'] for b in batch]
        hour_idx = [b['hour_idx'] for b in batch]
        complete_patient_stay = [b['complete_patient_stay'] for b in batch]
        full_patient_visit = [b['full_patient_visit'] for b in batch]
        prior_patient_state = [b['prior_patient_state'] for b in batch]
        prior_time = [b['prior_time'] for b in batch]

        prior_development = [restrict_patient_until_hour(prior_patient_state[i], hour_idx[i], ed_stay=True, restrict_to_N_hours=24)[0] for i in range(len(batch))]

        if only_gt:
            continue


        just_admitted = [False]*len(batch)
        ed_stay = [True]*len(batch)

        batch_output = simulate_development(full_patient_visit=full_patient_visit, sample=sample, stay_id=stay_id, model=model, tokenizer=tokenizer,
                                                                 model_name=model_name, hour_idx=hour_idx, complete_patient_stay=complete_patient_stay, prior_patient_state=prior_patient_state, prior_time=prior_time,
                                                                 icu_only=False, just_admitted=just_admitted, ed_stay=ed_stay, only_ed=True, max_steps=24, batch_collator=batch_collator, use_summ=use_summ,
                                                                 summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only)

        for j in range(len(batch)):
            ed_disposition, ed_icd, disposition, released_from_icu, icd, new_hour_idx, new_patient_state = batch_output[j]

            if new_patient_state is None:
                for value_col in value_cols:
                    gt_labels[f"ed_{value_col}"].append((labels[j][value_col], labels_time_tolerant[j][value_col]))

                    if value_col == "vitalsigns":
                        prior_labels[f"ed_{value_col}"].append(prior_development[j].patient_ed.ed_vital)
                        pred_labels[f"ed_{value_col}"].append(None)

                    elif value_col == "medications":
                        prior_labels[f"ed_{value_col}"].append(prior_development[j].patient_ed.ed_pyxis)
                        pred_labels[f"ed_{value_col}"].append(None)

            else: #valid prediction
                # restrict new patient state to last 24 hours or until ED discharge
                new_patient_state, end_time = restrict_patient_until_hour(new_patient_state, new_hour_idx, ed_stay=True, restrict_to_N_hours=new_hour_idx-hour_idx[j])

                for value_col in value_cols:
                    gt_labels[f"ed_{value_col}"].append((labels[j][value_col], labels_time_tolerant[j][value_col]))

                    if value_col == "vitalsigns":
                        prior_labels[f"ed_{value_col}"].append(prior_development[j].patient_ed.ed_vital)
                        pred_labels[f"ed_{value_col}"].append(new_patient_state.patient_ed.ed_vital)

                    elif value_col == "medications":
                        prior_labels[f"ed_{value_col}"].append(prior_development[j].patient_ed.ed_pyxis)
                        pred_labels[f"ed_{value_col}"].append(new_patient_state.patient_ed.ed_pyxis)


        with open(f"{eval_results_folder}/ed_ts/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/ed_ts/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)
        with open(f"{eval_results_folder}/ed_ts/prior_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(prior_labels, f)

        # save the current iteration to know where to restart from
        with open(f"{eval_results_folder}/ed_ts/current_iter_{model_path.replace('/', '_')}_{split}.txt", "w") as f:
            f.write(str(i))

        print(f"Saved current gt and pred labels at iteration {i} of {len(dataloader)}")

    return gt_labels, pred_labels, prior_labels

def eval_icu_ts(tokenizer, model, split='val', only_gt=False, value_cols=["vitalsigns", "medications"], time_tolerance=5, use_summ=False, summ_model=None, summ_tokenizer=None, summ_collator=None, summ_only=False):
    eval_data_collator = CustomEvalCollator()
    eval_dataset = ICUTimeseriesDataset(None, split=split, value_cols=value_cols, time_tolerance=time_tolerance)
    base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if use_summ:
        batch_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['desc_emb', 'sum_mask'])
    else:
        batch_collator = base_collator

    batch_size = 5
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator)

    if os.path.exists(f"{eval_results_folder}/icu_ts/current_iter_{model_path.replace('/', '_')}_{split}.txt"):
        with open(f"{eval_results_folder}/icu_ts/current_iter_{model_path.replace('/', '_')}_{split}.txt", "r") as f:
            current_iter = int(f.read())

        gt_labels = pickle.load(open(f"{eval_results_folder}/icu_ts/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        pred_labels = pickle.load(open(f"{eval_results_folder}/icu_ts/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        prior_labels = pickle.load(open(f"{eval_results_folder}/icu_ts/prior_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))

        print("Starting from iteration", current_iter)

    else:
        current_iter = -1
        gt_labels = defaultdict(list)
        pred_labels = defaultdict(list)
        prior_labels = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if batch == [{}]:
            continue
        if i <= current_iter:
            continue

        sample = [b['sample'] for b in batch]
        labels = [b['label'] for b in batch]
        labels_time_tolerant = [b['label_time_tolerant'] for b in batch]
        stay_id = [b['stay_id'] for b in batch]
        hour_idx = [b['hour_idx'] for b in batch]
        complete_patient_stay = [b['complete_patient_stay'] for b in batch]
        full_patient_visit = [b['full_patient_visit'] for b in batch]
        prior_patient_state = [b['prior_patient_state'] for b in batch]
        prior_time = [b['prior_time'] for b in batch]

        if only_gt:
            continue

        prior_development = [restrict_patient_until_hour(prior_patient_state[i], hour_idx[i], ed_stay=False, restrict_to_N_hours=24)[0] for i in range(len(batch))]
        just_admitted = [False]*len(batch)
        ed_stay = [False]*len(batch)

        batch_outputs = simulate_development(full_patient_visit=full_patient_visit, sample=sample, stay_id=stay_id, model=model, tokenizer=tokenizer,
                                                                 model_name=model_name, hour_idx=hour_idx, complete_patient_stay=complete_patient_stay, prior_patient_state=prior_patient_state, prior_time=prior_time,
                                                                 icu_only=True, just_admitted=just_admitted, ed_stay=ed_stay, only_ed=False, max_steps=24, batch_collator=batch_collator, use_summ=use_summ,
                                                                 summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only)


        for j in range(len(batch)):
            ed_disposition, ed_icd, disposition, released_from_icu, icd, new_hour_idx, new_patient_state = batch_outputs[j]

            if new_patient_state is None:
                for value_col in value_cols:
                    gt_labels[f"icu_{value_col}"].append((labels[j][value_col], labels_time_tolerant[j][value_col]))

                    if value_col == "vitalsigns":
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_icu[-1].chartevents['RoutineVitalSigns'])
                        pred_labels[f"icu_{value_col}"].append(None)

                    elif value_col == "medications":
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_icu[-1].inputevents)
                        pred_labels[f"icu_{value_col}"].append(None)

            else:  # valid prediction

                # restrict new patient state to last 24 hours or until discharge
                new_patient_state, end_time = restrict_patient_until_hour(new_patient_state, new_hour_idx, ed_stay=False, restrict_to_N_hours=new_hour_idx-hour_idx[j])

                for value_col in value_cols:
                    gt_labels[f"icu_{value_col}"].append((labels[j][value_col], labels_time_tolerant[j][value_col]))

                    if value_col == "vitalsigns": # changes always refer to the most recent icu stay
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_icu[-1].chartevents['RoutineVitalSigns'])
                        pred_labels[f"icu_{value_col}"].append(new_patient_state.patient_icu[-1].chartevents['RoutineVitalSigns']) if 'RoutineVitalSigns' in new_patient_state.patient_icu[-1].chartevents else None

                    elif value_col == "medications":
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_icu[-1].inputevents)
                        pred_labels[f"icu_{value_col}"].append(new_patient_state.patient_icu[-1].inputevents)


        with open(f"{eval_results_folder}/icu_ts/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/icu_ts/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)
        with open(f"{eval_results_folder}/icu_ts/prior_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(prior_labels, f)

        with open(f"{eval_results_folder}/icu_ts/current_iter_{model_path.replace('/', '_')}_{split}.txt", "w") as f:
            f.write(str(i))

        print(f"Saved current gt and pred labels at iteration {i} of {len(dataloader)}")

    return gt_labels, pred_labels, prior_labels

def eval_hosp_ts(tokenizer, model, split='val', only_gt=False, value_cols=["medications"], time_tolerance=5, use_summ=False, summ_model=None, summ_tokenizer=None, summ_collator=None, summ_only=False):
    eval_data_collator = CustomEvalCollator()
    eval_dataset = HospTimeseriesDataset(None, split=split, value_cols=value_cols, time_tolerance=time_tolerance)

    base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    if use_summ:
        batch_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['desc_emb', 'sum_mask'])
    else:
        batch_collator = base_collator

    batch_size = 8
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator)

    if os.path.exists(f"{eval_results_folder}/hosp_ts_labs/current_iter_{model_path.replace('/', '_')}_{split}.txt"):
        with open(f"{eval_results_folder}/hosp_ts_labs/current_iter_{model_path.replace('/', '_')}_{split}.txt", "r") as f:
            current_iter = int(f.read())

        gt_labels = pickle.load(open(f"{eval_results_folder}/hosp_ts_labs/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        pred_labels = pickle.load(open(f"{eval_results_folder}/hosp_ts_labs/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        prior_labels = pickle.load(open(f"{eval_results_folder}/hosp_ts_labs/prior_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "rb"))
        prior_times = defaultdict(list)

    else:
        current_iter = -1
        gt_labels = defaultdict(list)
        pred_labels = defaultdict(list)
        prior_labels = defaultdict(list)
        prior_times = defaultdict(list)

    for i, batch in tqdm(enumerate(dataloader)):
        if batch == [{}]:
            continue
        if i <= current_iter:
            continue

        sample = [b['sample'] for b in batch]
        labels = [b['label'] for b in batch]
        labels_time_tolerant = [b['label_time_tolerant'] for b in batch]
        stay_id = [b['stay_id'] for b in batch]
        hour_idx = [b['hour_idx'] for b in batch]
        complete_patient_stay = [b['complete_patient_stay'] for b in batch]
        full_patient_visit = [b['full_patient_visit'] for b in batch]
        prior_patient_state = [b['prior_patient_state'] for b in batch]
        prior_time = [b['prior_time'] for b in batch]
        prior_development = [restrict_patient_until_hour(prior_patient_state[i], hour_idx[i], ed_stay=False, restrict_to_N_hours=24)[0] for i in range(len(batch))]

        if only_gt:
            for j in range(len(batch)):
                for value_col in value_cols:
                    gt_labels[f"icu_{value_col}"].append((labels[j][value_col], labels_time_tolerant[j][value_col]))
                    prior_times[f"icu_{value_col}"].append(prior_time[j])
            continue

        just_admitted = [False]*len(batch)
        ed_stay = [False]*len(batch)

        batch_outputs = simulate_development(full_patient_visit=full_patient_visit, sample=sample, stay_id=stay_id, model=model, tokenizer=tokenizer,
                                                                 model_name=model_name, hour_idx=hour_idx, complete_patient_stay=complete_patient_stay, prior_patient_state=prior_patient_state, prior_time=prior_time,
                                                                 icu_only=False, just_admitted=just_admitted, ed_stay=ed_stay, only_ed=False, max_steps=24, batch_collator=batch_collator, use_summ=use_summ,
                                                                 summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only)

        for j in range(len(batch)):
            ed_disposition, ed_icd, disposition, released_from_icu, icd, new_hour_idx, new_patient_state = batch_outputs[j]

            if new_patient_state is None:
                for value_col in value_cols:
                    gt_labels[f"icu_{value_col}"].append((labels[j][value_col], labels_time_tolerant[j][value_col]))

                    if value_col == "vitalsigns":
                        raise ValueError("Vitalsigns not supported for hospitalization task")

                    elif value_col == "medications":
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_adm.prescriptions)
                        pred_labels[f"icu_{value_col}"].append(None)

                    elif value_col == "labs":
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_adm[-1].labevents)
                        pred_labels[f"icu_{value_col}"].append(None)

            else:  # valid prediction

                # restrict new patient state to last 24 hours or until discharge
                new_patient_state, end_time = restrict_patient_until_hour(new_patient_state, new_hour_idx, ed_stay=False, restrict_to_N_hours=new_hour_idx-hour_idx[j])

                for value_col in value_cols:
                    gt_labels[f"icu_{value_col}"].append((labels[j][value_col], labels_time_tolerant[j][value_col]))

                    if value_col == "vitalsigns":
                        raise ValueError("Vitalsigns not supported for hosp_ts")

                    elif value_col == "medications":
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_adm.prescriptions)
                        pred_labels[f"icu_{value_col}"].append(new_patient_state.patient_adm.prescriptions)

                    elif value_col == "labs":
                        prior_labels[f"icu_{value_col}"].append(prior_development[j].patient_adm.labevents)
                        pred_labels[f"icu_{value_col}"].append(new_patient_state.patient_adm.labevents)


        with open(f"{eval_results_folder}/hosp_ts_labs/gt_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/hosp_ts_labs/pred_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)
        with open(f"{eval_results_folder}/hosp_ts_labs/prior_labels_{model_path.replace('/', '_')}_curr_{split}.pkl", "wb") as f:
            pickle.dump(prior_labels, f)

        with open(f"{eval_results_folder}/hosp_ts_labs/current_iter_{model_path.replace('/', '_')}_{split}.txt", "w") as f:
            f.write(str(i))

        print(f"Saved current gt and pred labels at iteration {i} of {len(dataloader)}")

    return gt_labels, pred_labels, prior_labels, prior_times

def load_model(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="cuda"
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    if use_summarization:
        new_special_tokens = {'additional_special_tokens': ['<SUMMARY>']}
        tokenizer.add_special_tokens(new_special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # file_path = f"/home/guests/chantal_pellegrini/Holistic_Patient_Pathway/outputs/train_qwen_5000_v2_weight_24h_1M_withICU/checkpoint-120000/adapter_model.safetensors"
    # file_path = "/home/guests/chantal_pellegrini/Holistic_Patient_Pathway/outputs/train_qwen_5000_v2_weight_24h_1M_withICU_withLOSLeak/checkpoint-60000/adapter_model.safetensors"
    file_path = f"{model_path}adapter_model.safetensors"
    lora_weights = load_file(file_path)
    # change all keys from .weight to .default.weight
    lora_weights = {k.replace(".weight", ".default.weight"): v for k, v in lora_weights.items()}
    model.load_state_dict(lora_weights, strict=False)

    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Load config for training")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--use_los' , action='store_true')
    parser.add_argument('--use_summ', action='store_true')
    parser.add_argument('--summ_only', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    # get model path from command line arguments
    args = parse_args()
    model_path = args.model_path
    use_los = args.use_los
    use_summ = args.use_summ
    task = args.task
    split = args.split
    summ_only = args.summ_only

    model, tokenizer = load_model(model_path = model_path)
    if use_summ:
        print("Loading summarization model")
        # load summarization model
        summ_model, summ_tokenizer = load_summ_model()
        # move to cpu
        summ_model = summ_model.cpu()
        base_collator = transformers.DataCollatorForSeq2Seq(
            summ_tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

        summ_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['stay_id', 'hour_idx'])
    else:
        summ_model, summ_tokenizer, summ_collator = None, None, None

    tasks = [task]
    # save metrics to text file

    if "Discharge_ICD" in tasks:
        print("Discharge ICD")
        gt_labels, pred_labels = eval_icd(tokenizer, model, split=split, only_gt=False, use_los=True, use_summ=use_summ,
                                          summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only) # always use input_los (0 hours) as we want to predict diagnosis at discharge
        # save gt_labels and pred_labels
        # create folder if it does not exist
        if not os.path.exists(f"{eval_results_folder}/discharge_icd"):
            os.makedirs(f"{eval_results_folder}/discharge_icd")

        with open(f"{eval_results_folder}/discharge_icd/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/discharge_icd/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)

        # calculate multi-label multi-class metrics
        metrics = calculate_multilabel_metrics(gt_labels, pred_labels)
        print(metrics)
        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/discharge_icd/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.close()

    if "ED_ADM" in tasks:
        print("ED Admission")
        gt_labels, pred_labels = eval_ed(tokenizer, model, split=split, only_gt=False, at_discharge=False, use_summ=use_summ,
                                         summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only, predict_adm=True,
                                         eval_future_ed_disp = '_direct' in model_path)

        if not os.path.exists(f"{eval_results_folder}/ed_adm"):
            os.makedirs(f"{eval_results_folder}/ed_adm")
        with open(f"{eval_results_folder}/ed_adm/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/ed_adm/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)

        metrics = calculate_metrics({k: v for k, v in gt_labels.items() if k != "ICD"}, {k: v for k, v in pred_labels.items() if k != "ICD"})
        icd_metrics = calculate_multilabel_metrics({k: v for k, v in gt_labels.items() if k == "ICD"}, {k: v for k, v in pred_labels.items() if k == "ICD"})
        print(metrics)
        print(icd_metrics)
        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/ed_adm/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.write(str(icd_metrics))
        metrics_file.close()

    if "ED_ICD" in tasks:
        print("ED ICD Prediction at discharge")
        gt_labels, pred_labels = eval_ed(tokenizer, model, split=split, only_gt=False, at_discharge=True, use_los=True, use_summ=use_summ,
                                         summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only, predict_adm=False)

        if not os.path.exists(f"{eval_results_folder}/ed_icd"):
            os.makedirs(f"{eval_results_folder}/ed_icd")
        with open(f"{eval_results_folder}/ed_icd/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/ed_icd/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)

        metrics = calculate_metrics({k: v for k, v in gt_labels.items() if k != "ICD"}, {k: v for k, v in pred_labels.items() if k != "ICD"})
        icd_metrics = calculate_multilabel_metrics({k: v for k, v in gt_labels.items() if k == "ICD"}, {k: v for k, v in pred_labels.items() if k == "ICD"})
        print(metrics)
        print(icd_metrics)
        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/ed_icd/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.write(str(icd_metrics))
        metrics_file.close()

    if "ICU_LOS" in tasks:
        print("ICU LOS")
        if not os.path.exists(f"{eval_results_folder}/icu_los"):
            os.makedirs(f"{eval_results_folder}/icu_los")
        gt_labels, pred_labels = eval_ICU(tokenizer, model, split=split, only_gt=False, use_los=use_los, use_summ=use_summ, summ_model=summ_model,
                                          summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only,
                                          eval_future_los='_direct' in model_path, model_path=model_path)

        with open(f"{eval_results_folder}/icu_los/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/icu_los/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)

        metrics = calculate_metrics(gt_labels, pred_labels)
        # aurocs = calculate_auroc(eval_ICU, tokenizer, model, split=split) # should also work for all other tasks but did not try it yet
        print(metrics)
        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/icu_los/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.close()

    if "ICU_Mort" in tasks:
        print("ICU 24h Imminent Mortality")
        if not os.path.exists(f"{eval_results_folder}/icu_mort"):
            os.makedirs(f"{eval_results_folder}/icu_mort")
        gt_labels, pred_labels = eval_ICU_mort(tokenizer, model, split=split, only_gt=False, use_summ=use_summ, summ_model=summ_model, summ_tokenizer=summ_tokenizer,
                                               summ_collator=summ_collator, summ_only=summ_only, eval_future_mort='_direct' in model_path, model_path=model_path)

        with open(f"{eval_results_folder}/icu_mort/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/icu_mort/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)

        metrics = calculate_metrics(gt_labels, pred_labels)
        # aurocs = calculate_auroc(eval_ICU, tokenizer, model, split=split) # should also work for all other tasks but did not try it yet
        print(metrics)
        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/icu_mort/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.close()

    if "ED_TS" in tasks:
        time_tolerance = 1

        if not os.path.exists(f"{eval_results_folder}/ed_ts"):
            os.makedirs(f"{eval_results_folder}/ed_ts")

        gt_labels, pred_labels, prior_labels = eval_ed_ts(tokenizer, model, split=split, only_gt=False, value_cols=["vitalsigns", "medications"],
                                                          time_tolerance=time_tolerance, use_summ=use_summ, summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only)

        with open(f"{eval_results_folder}/ed_ts/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(gt_labels, f)
        with open(f"{eval_results_folder}/ed_ts/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(pred_labels, f)
        with open(f"{eval_results_folder}/ed_ts/prior_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
            pickle.dump(prior_labels, f)

        metrics = calculate_ts_metrics(gt_labels, pred_labels, time_tolerance=time_tolerance)
        print(metrics)

        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/ed_ts/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.close()

    if "ICU_TS" in tasks:
        time_tolerance = 1

        if not os.path.exists(f"{eval_results_folder}/icu_ts"):
            os.makedirs(f"{eval_results_folder}/icu_ts")
        # split = f"_{split}"

        if os.path.exists(f"{eval_results_folder}/icu_ts/gt_labels_{model_path.replace('/', '_')}_{split}.pkl"):
            gt_labels = pickle.load(open(f"{eval_results_folder}/icu_ts/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "rb"))
            pred_labels = pickle.load(open(f"{eval_results_folder}/icu_ts/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "rb"))
            prior_labels = pickle.load(open(f"{eval_results_folder}/icu_ts/prior_labels_{model_path.replace('/', '_')}_{split}.pkl", "rb"))

        else:

            gt_labels, pred_labels, prior_labels = eval_icu_ts(tokenizer, model, split=split, only_gt=False, value_cols=["vitalsigns", "medications"], time_tolerance=time_tolerance,
                                                               use_summ=use_summ, summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only)

            with open(f"{eval_results_folder}/icu_ts/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
                pickle.dump(gt_labels, f)
            with open(f"{eval_results_folder}/icu_ts/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
                pickle.dump(pred_labels, f)
            with open(f"{eval_results_folder}/icu_ts/prior_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
                pickle.dump(prior_labels, f)

        metrics = calculate_ts_metrics(gt_labels, pred_labels, time_tolerance=time_tolerance)
        print(metrics)

        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/icu_ts/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.close()

    if "HOSP_TS" in tasks:
        time_tolerance = 1
        if not os.path.exists(f"{eval_results_folder}/hosp_ts_labs"):
            os.makedirs(f"{eval_results_folder}/hosp_ts_labs")

        if os.path.exists(f"{eval_results_folder}/hosp_ts_labs/pred_labels_{model_path.replace('/', '_')}_{split}.pkl"):
            gt_labels = pickle.load(open(f"{eval_results_folder}/hosp_ts_labs/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "rb"))
            pred_labels = pickle.load(open(f"{eval_results_folder}/hosp_ts_labs/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "rb"))
            prior_labels = pickle.load(open(f"{eval_results_folder}/hosp_ts_labs/prior_labels_{model_path.replace('/', '_')}_{split}.pkl", "rb"))
            prior_times = None

        else:
            gt_labels, pred_labels, prior_labels, prior_times = eval_hosp_ts(tokenizer, model, split=split, only_gt=False, value_cols=["labs"], time_tolerance=time_tolerance,
                                                                use_summ=use_summ, summ_model=summ_model, summ_tokenizer=summ_tokenizer, summ_collator=summ_collator, summ_only=summ_only)

            with open(f"{eval_results_folder}/hosp_ts_labs/gt_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
                pickle.dump(gt_labels, f)
            with open(f"{eval_results_folder}/hosp_ts_labs/pred_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
                pickle.dump(pred_labels, f)
            with open(f"{eval_results_folder}/hosp_ts_labs/prior_labels_{model_path.replace('/', '_')}_{split}.pkl", "wb") as f:
                pickle.dump(prior_labels, f)

        metrics = calculate_ts_metrics(gt_labels, pred_labels, time_tolerance=time_tolerance)
        print(metrics)

        # save metrics to text file
        metrics_file = open(f"{eval_results_folder}/hosp_ts_labs/metrics_{model_path.replace('/', '_')}_{split}.txt", "a+")
        metrics_file.write(str(metrics))
        metrics_file.close()