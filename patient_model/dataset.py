import atexit
import copy
import gzip
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import transformers
import yaml
from torch.utils.data import Dataset, get_worker_info, DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm
import torch.multiprocessing as mp

from transformers import AutoTokenizer

from mimic_iv_extraction.paths import mimiciv_path as mimiciv_path
from patient_model.retrieve_patient_model import summerize_patient_lvl_4, summerize_patient_lvl_4_summary_data, split_section_into_parts

# my_pool = mp.Pool(processes=1)
# atexit.register(lambda: my_pool.close() or my_pool.join())  # Ensure cleanup at exit
# print("Pool created")

def create_custom_attention_mask(input_len, sum_len, output_len):
    """
    Creates a custom attention mask for the input, summary, and output tokens.

    Parameters:
    - input_len: Number of input tokens
    - sum_len: Number of summery tokens
    - output_len: Number of output tokens

    Returns:
    - mask: 4D torch tensor of shape [1, seq_len, seq_len] representing the attention mask
    """
    seq_len = input_len + sum_len + 3 + output_len
    mask = np.zeros((seq_len, seq_len), dtype=float)

    # Indices for different segments
    input_end = input_len
    sum_start = input_end
    sum_end = input_end + sum_len
    output_start = sum_end

    # Classic causal mask up to the last SUM token
    mask[:sum_end, :sum_end] = np.tril(np.ones((sum_end, sum_end)))

    # Output tokens can only attend to SUM tokens and themselves
    for i in range(output_start, seq_len):
        mask[i, sum_start:i + 1] = 1  # Allow attending to only SUM tokens and previous output tokens

    # Convert to the proper attention mask format (large negative values for masking out)
    return torch.tensor(mask).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len] - batch size and head dimensions


def generate_and_tokenize_prompt(input, output, tokenizer, train_on_inputs=False, predict=False, model_name="Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                                 custom_attn_mask=False, add_gen_prompt_for_predict=True, add_summary_tokens=None, gen_summ_mask=False, summary_len=8):
    SUMMARY_LENGTH = summary_len
    CONV_TOKENS_AFTER_SUMMARY = 5
    if add_summary_tokens is None:
        add_summary_tokens = custom_attn_mask
    if add_summary_tokens:
        # add 10 summary tokens ('<SUMMARY>') to input
        input = input + "<SUMMARY>" * SUMMARY_LENGTH

    if predict:  # for prediction, we don't have the output

        if model_name == "Meta-Llama-3.1-8B-Instruct-bnb-4bit":
            messages = [
                {"role": "system", "content": "You are a bot for predicting the changes in a patient state in the next hour."},
                {"role": "user", "content": input}
            ]
        else:  # qwen
            messages = [
                {"role": "user", "content": input},
                # {"role": "assistant", "content": output[:93]} #activate to test "prompting" the model, in generate input need to delete last two tokens from input and attn_mask
            ]

        tokenized_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=add_gen_prompt_for_predict, return_dict=True)
        tokenized_labels = tokenizer(output, add_special_tokens=False)['input_ids']
        tokenized_prompt["labels"] = tokenized_labels

        if gen_summ_mask:
            summ_mask = [1 if token == tokenizer.added_tokens_encoder['<SUMMARY>'] else 0 for token in tokenized_prompt["input_ids"]]
            return tokenized_prompt, summ_mask

        return tokenized_prompt

    if model_name == "Meta-Llama-3.1-8B-Instruct-bnb-4bit":
        messages_full = [
            {"role": "system", "content": "You are a bot for predicting the changes in a patient state in the next hour."},
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ]
    else:  # qwen
        messages_full = [
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ]

    tokenized_full_prompt = tokenizer.apply_chat_template(messages_full, add_generation_prompt=False, return_dict=True)

    if not train_on_inputs:
        user_messages = messages_full[:-1]
        tokenized_user_prompt = tokenizer.apply_chat_template(user_messages, add_generation_prompt=True, return_dict=False)
        user_prompt_len = len(tokenized_user_prompt)

        tokenized_full_prompt.data["labels"] = [
                                                   -100
                                               ] * user_prompt_len + tokenized_full_prompt["input_ids"][
                                                                     user_prompt_len:
                                                                     ]

        if custom_attn_mask:
            attn_mask = create_custom_attention_mask(user_prompt_len - SUMMARY_LENGTH - CONV_TOKENS_AFTER_SUMMARY, SUMMARY_LENGTH, len(
                tokenized_full_prompt["input_ids"]) - user_prompt_len)  # 5 for special tokens (<|im_end|>, <|im_start|>, assistant)
            tokenized_full_prompt.data["attention_mask"] = attn_mask
    if gen_summ_mask:
        summ_mask = [1 if token == tokenizer.added_tokens_encoder['<SUMMARY>'] else 0 for token in tokenized_full_prompt["input_ids"]]
        return tokenized_full_prompt, summ_mask

    return tokenized_full_prompt


def generate_and_tokenize_prompt_full_summary(input, output, tokenizer, train_on_inputs=False, predict=False, model_name="Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                                              sum_emb_dict=None, summary_len=8, drop_augment = False, eval_mode='', summ_ratio=0.33):
    SUMMARY_LENGTH = summary_len

    input_dict = {}
    sum_string = "<SUMMARY>" * SUMMARY_LENGTH
    for key, sum_emb in sum_emb_dict.items():
        section_keys = sections[key[3]]
        stay = key[2]
        is_icu = section_keys[0][0] == "ICU"
        if is_icu:
            if len(section_keys[0]) == 2:
                input_dict.setdefault("ICU Stay", {}).setdefault(stay, {}).setdefault(section_keys[0][1], "")
                input_dict["ICU Stay"][stay][section_keys[0][1]] += sum_string
            elif len(section_keys[0]) == 3:
                input_dict.setdefault("ICU Stay", {}).setdefault(stay, {}).setdefault(section_keys[0][1], {}).setdefault(section_keys[0][2], "")
                input_dict["ICU Stay"][stay][section_keys[0][1]][section_keys[0][2]] += sum_string
        else:
            if len(section_keys[0]) == 1:
                input_dict.setdefault(section_keys[0][0], "")
                input_dict[section_keys[0][0]] += sum_string
            elif len(section_keys[0]) == 2:
                input_dict.setdefault(section_keys[0][0], {}).setdefault(section_keys[0][1], "")
                input_dict[section_keys[0][0]][section_keys[0][1]] += sum_string

    input_sum = yaml.dump(input_dict, sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
    if drop_augment: # started with 0.33 for all, now testing if more summary only is better as model still not seems to be able to use summary
        prob = random.random()
        if summ_ratio == 0.33:
            if prob < 0.33:
                # keep only summary
                input = "Summary: \n" + input_sum
            elif prob < 0.66:
                # keep only text
                input = "Recent: \n" + input
            else:
                # keep both
                input = "Summary: \n" + input_sum + "Recent: \n" + input

        elif summ_ratio == 0.5:
            if prob < 0.5:
                # keep only summary
                input = "Summary: \n" + input_sum
            elif prob < 0.75:
                # keep only text
                input = "Recent: \n" + input
            else:
                # keep both
                input = "Summary: \n" + input_sum + "Recent: \n" + input

        elif summ_ratio == 0.66:
            if prob < 0.66:
                # keep only summary
                input = "Summary: \n" + input_sum
            elif prob < 0.83:
                # keep only text
                input = "Recent: \n" + input
            else:
                # keep both
                input = "Summary: \n" + input_sum + "Recent: \n" + input

    # used to evaluate mixed model (trained with dropping) on only summary or only text - during training drop_augment should be True
    elif eval_mode != '':
        if eval_mode == 'summary':
            input = "Summary: \n" + input_sum
        elif eval_mode == 'text':
            input = "Recent: \n" + input
        elif eval_mode == 'both':
            input = "Summary: \n" + input_sum + "Recent: \n" + input

    else:
        input = "Summary: \n" + input_sum + "Recent: \n" + input

    if predict:  # for prediction, we don't have the output

        if model_name == "Meta-Llama-3.1-8B-Instruct-bnb-4bit":
            messages = [
                {"role": "system", "content": "You are a bot for predicting the changes in a patient state in the next hour."},
                {"role": "user", "content": input}
            ]
        else:  # qwen
            messages = [
                {"role": "user", "content": input},
                # {"role": "assistant", "content": output[:93]} #activate to test "prompting" the model, in generate input need to delete last two tokens from input and attn_mask
            ]

        tokenized_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True) #add_generation_prompt correct?
        tokenized_labels = tokenizer(output, add_special_tokens=False)['input_ids']
        tokenized_prompt["labels"] = tokenized_labels

        summ_mask = [1 if token == tokenizer.added_tokens_encoder['<SUMMARY>'] else 0 for token in tokenized_prompt["input_ids"]]
        return tokenized_prompt, summ_mask

    if model_name == "Meta-Llama-3.1-8B-Instruct-bnb-4bit":
        messages_full = [
            {"role": "system", "content": "You are a bot for predicting the changes in a patient state in the next hour."},
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ]
    else:  # qwen
        messages_full = [
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ]

    tokenized_full_prompt = tokenizer.apply_chat_template(messages_full, add_generation_prompt=False, return_dict=True)

    if not train_on_inputs:
        user_messages = messages_full[:-1]
        tokenized_user_prompt = tokenizer.apply_chat_template(user_messages, add_generation_prompt=True, return_dict=False)
        user_prompt_len = len(tokenized_user_prompt)

        tokenized_full_prompt.data["labels"] = [
                                                   -100
                                               ] * user_prompt_len + tokenized_full_prompt["input_ids"][
                                                                     user_prompt_len:
                                                                     ]


    summ_mask = [1 if token == tokenizer.added_tokens_encoder['<SUMMARY>'] else 0 for token in tokenized_full_prompt["input_ids"]]
    return tokenized_full_prompt, summ_mask


''' Dataset for text-only next step models: output = tokenized text prompts -> next step '''
class TextDataset(Dataset):
    def __init__(self, tokenizer, split='train', predict=False, max_input_len=4000, max_output_len=1000,
                 model_name="Meta-Llama-3.1-8B-Instruct-bnb-4bit", weighted_sampling=False, custom_attn_mask=False, dataset_name="24h_los",
                 predict_all=False, add_gen_prompt_for_predict=True):
        self.add_gen_prompt_for_predict = add_gen_prompt_for_predict
        # list files in all_data
        if weighted_sampling == True and split == "train":
            with gzip.open(f"{mimiciv_path}{split}_weighting_sample_timepoint_paths_1Mio.json.gz", "rt") as f:
                self.data = json.load(f)
        else:  # for validation and test set we don't want to use weighted sampling, but actual data distribution
            with gzip.open(f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz", "rt") as f:
                self.data = json.load(f)
            self.data = self.data[:8000]
        print(f"Loaded {len(self.data)} samples for {split} split")

        if predict and not predict_all:  # for generation and metric calculation, we only need a subset of the data, as it takes too long otherwise
            # select 200 random, but fixed samples for qualitative evaluation using sample
            # random.seed(42)
            # self.data = random.sample(self.data, 200)
            self.data = self.data[:5000] # 5000 for testing, in training used 500
            print(f"Predicting on {len(self.data)} samples.")

        self.tokenizer = tokenizer
        self.split = split
        self.predict = predict

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.model_name = model_name
        self.custom_attn_mask = custom_attn_mask
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        if self.dataset_name == "24h_los":
            sample = json.load(open(mimiciv_path + "all_data_24_hours_los_noisy/" + sample_path, "r"))
        elif self.dataset_name == "all_steps":
            sample = json.load(open(mimiciv_path + "all_data_summarylvl1_los_noisy/" + sample_path, "r"))
        else:
            raise ValueError("Invalid dataset_name")

        try:
            sample = sample[timepoint_idx]
        except Exception:
            sample = sample[int(timepoint_idx)]

        # dynamically adjust input and output length
        sample["desc"] = summerize_patient_lvl_4(sample["desc"], tokenizer=self.tokenizer, max_num_tokens=self.max_input_len)
        sample["change_log"] = summerize_patient_lvl_4(sample["change_log"], tokenizer=self.tokenizer, max_num_tokens=self.max_output_len,
                                                       changelog=True)

        yaml.add_representer(float, self.float_representer, Dumper=yaml.SafeDumper)
        input = yaml.dump(sample["desc"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
        output = yaml.dump(sample["change_log"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)

        prompt = generate_and_tokenize_prompt(input, output, self.tokenizer, predict=self.predict, model_name=self.model_name,
                                              custom_attn_mask=self.custom_attn_mask, gen_summ_mask=False,
                                              add_gen_prompt_for_predict=self.add_gen_prompt_for_predict)

        if self.predict == False:
            return prompt
        else:
            return {
                'input_ids': prompt['input_ids'],
                'labels': prompt['labels'],
                'attention_mask': prompt['attention_mask'],
                'stay_id': sample["stay_id"],
                'hour_idx': sample["hour_idx"]
            }


def get_section_info(sample, sample_info, section_name, section_key, sub_keys=None, sub_sub_keys=None, stay=""):
    static_sections = ['HS_General__HA_PatientLocation__HA_CareTaker', 'HS_OutpatientMeasurements', 'HS_RadiologyNotes', 'ICU_CE_MDProgressNote',
                       'ICU_CE_AdmHistory']

    # Handle ICU stays if a specific stay is provided
    if stay != "":
        desc = sample.get("desc", {}).get(section_key, {}).get(stay, {})
    else:
        # For non-ICU sections or where stay is not relevant
        desc = sample.get("desc", {}).get(section_key, {})
    # Access deeper levels if needed
    if sub_keys:
        # merge sections of all sub_keys
        desc = {sub_key: desc.get(sub_key, {}) for sub_key in sub_keys if desc.get(sub_key, {}) != {}}
    if sub_sub_keys:
        # merge sections of all sub_sub_keys
        desc = {sub_sub_key: desc.get(sub_keys[0], {}).get(sub_sub_key, {}) for sub_sub_key in sub_sub_keys if
                desc.get(sub_keys[0], {}).get(sub_sub_key, {}) != {}}

    if desc == {}:
        print(f"Section {section_name} is empty for sample {sample_info}.")
        return None, None
    # assert desc != {}, f"Section {section_name} is empty for sample {sample_info}."
    # Determine if there is any content in desc
    if section_name in static_sections:
        if stay != "":
            desc = {section_key: {stay: desc}}
        else:
            desc = {section_key: desc}
        return desc, None

    # Determine if section has changed or is empty in the change log
    if stay != "":
        change_log = sample.get("change_log", {}).get(section_key, {}).get(stay, {})
    else:
        change_log = sample.get("change_log", {}).get(section_key, {})

    if sub_keys:
        change_log = {sub_key: change_log.get(sub_key, {}) for sub_key in sub_keys}
    if sub_sub_keys:
        change_log = {sub_sub_key: change_log.get(sub_keys[0], {}).get(sub_sub_key, {}) for sub_sub_key in sub_sub_keys}

    if stay != "":
        if 'Chart Events' in sub_keys:
            desc = {section_key: {stay: {'Chart Events': desc}}}
            change_log = {section_key: {stay: {'Chart Events': change_log}}}
        else:
            desc = {section_key: {stay: desc}}
            change_log = {section_key: {stay: change_log}}
    else:
        desc = {section_key: desc}
        change_log = {section_key: change_log}

    return desc, change_log


sections = {'ED_Stay': [('Emergency Department Stay',)],
            'HS_General__HA_PatientLocation__HA_CareTaker': [('Hospital Stay', 'General'), ('Hospital Stay', 'Patient Location'),
                                                             ('Hospital Stay', 'Care Taker')],
            'HS_OutpatientMeasurements': [('Hospital Stay', 'Outpatient Measurements')],
            'HS_LabResults': [('Hospital Stay', 'Lab Results')],
            'HS_RadiologyNotes': [('Hospital Stay', 'Radiology Notes')],
            'HS_Prescriptions': [('Hospital Stay', 'Prescriptions')],
            'HS_Procedures': [('Hospital Stay', 'Procedures')],
            'HS_Microbiology': [('Hospital Stay', 'Microbiology Growth Results')],
            'ICU_Medication': [('ICU', 'Medication')],
            'ICU_Output': [('ICU', 'Output')],
            'ICU_Procedures': [('ICU', 'Procedures')],
            'ICU_CE_VitalSigns': [('ICU', 'Chart Events', 'RoutineVitalSigns')],
            'ICU_CE_AdmHistory': [('ICU', 'Chart Events', 'AdmHistory_FHPA')],
            'ICU_CE_MDProgressNote': [('ICU', 'Chart Events', 'MDProgressNote')],
            'ICU_CE_Respiratory': [('ICU', 'Chart Events', 'Respiratory')],
            'ICU_CE_Pulmonary': [('ICU', 'Chart Events', 'Pulmonary')],
            'ICU_CE_SkinAssessment': [('ICU', 'Chart Events', 'Skin-Assessment')],
            'ICU_CE_SkinImpairment': [('ICU', 'Chart Events', 'Skin-Impairment')],
            'ICU_CE_SkinIncisions': [('ICU', 'Chart Events', 'Skin-Incisions')],
            'ICU_CE_CardioPulses': [('ICU', 'Chart Events', 'Cardiovascular(Pulses)')],
            'ICU_CE_Neurological': [('ICU', 'Chart Events', 'Neurological')],
            'ICU_CE_Hemodynamics': [('ICU', 'Chart Events', 'Hemodynamics')],
            'ICU_CE_Alarms': [('ICU', 'Chart Events', 'Alarms')],
            'ICU_CE_PainSedation': [('ICU', 'Chart Events', 'Pain_Sedation')],
            'ICU_CE_GIGU': [('ICU', 'Chart Events', 'GI_GU')],
            'ICU_CE_Cardio': [('ICU', 'Chart Events', 'Cardiovascular')],
            'ICU_CE_CardioPacerData': [('ICU', 'Chart Events', 'Cardiovascular(PacerData)')],
            'ICU_CE_IABP': [('ICU', 'Chart Events', 'IABP')],
            'ICU_CE_Dialysis': [('ICU', 'Chart Events', 'Dialysis')],
            'ICU_CE_Toxicology': [('ICU', 'Chart Events', 'Toxicology')],
            'ICU_CE_NICOM': [('ICU', 'Chart Events', 'NICOM')],
            }

''' Dataset for Training Section Summary models: output = tokenized single sections -> next step per section '''
class SummaryDataset(Dataset):
    def __init__(self, tokenizer, split='train', predict=False, max_input_len=1000, max_output_len=500,
                 model_name="Meta-Llama-3.1-8B-Instruct-bnb-4bit", custom_attn_mask=True, add_gen_prompt_for_predict=True,
                 predict_all=False, summary_len=8):

        self.add_gen_prompt_for_predict = add_gen_prompt_for_predict
        self.summary_len = summary_len
        with open(mimiciv_path + f"section_samples_for_summary_{split}.json", "r") as f:
            self.data = json.load(f)

        # sort data
        self.data = sorted(self.data)

        if predict and not predict_all:  # for generation and metric calculation, we only need a subset of the data, as it takes too long otherwise
            # select 200 random, but fixed samples for qualitative evaluation using sample
            # random.seed(42)
            # self.data = random.sample(self.data, min(len(self.data), 200))
            self.data = self.data[:5000]

        self.tokenizer = tokenizer
        self.split = split
        self.predict = predict

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.model_name = model_name
        self.custom_attn_mask = custom_attn_mask

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):

        elem = self.data[idx]
        sample_info_stay, section = elem
        sample_info, icu_stay = sample_info_stay
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        sample = json.load(open(mimiciv_path + "all_data_summarylvl1_los_noisy/" + sample_path, "r"))
        try:
            sample = sample[timepoint_idx]
        except Exception:
            try:
                sample = sample[int(timepoint_idx)]
            except Exception:
                print(f"Error loading sample {sample_path}")
                raise ValueError(f"Error loading sample {sample_path}")

        section_group = sections[section]

        is_icu = section_group[0][0] == "ICU"

        if is_icu:
            if len(section_group[0]) == 2:
                desc, change_log = get_section_info(sample, sample_info, section,
                                                    section_key="ICU Stay", sub_keys=[section[1] for section in section_group], stay=icu_stay)
            elif len(section_group[0]) == 3:
                desc, change_log = get_section_info(sample, sample_info, section,
                                                    section_key="ICU Stay", sub_keys=[section_group[0][1]],
                                                    sub_sub_keys=[section[2] for section in section_group], stay=icu_stay)
            else:
                raise ValueError("Invalid section group")

        else:
            if len(section_group[0]) == 1:
                desc, change_log = get_section_info(sample, sample_info, section, section_key=section_group[0][0])

            elif len(section_group[0]) == 2:
                desc, change_log = get_section_info(sample, sample_info, section, section_key=section_group[0][0],
                                                    sub_keys=[section[1] for section in section_group])
            else:
                raise ValueError("Invalid section group")

        sample["desc"] = desc
        sample["desc"] = summerize_patient_lvl_4_summary_data(sample["desc"], tokenizer=self.tokenizer,
                                                              max_num_tokens=self.max_input_len if change_log is not None else (self.max_output_len + self.max_input_len) // 2)

        if change_log is None:
            sample["change_log"] = sample["desc"]
        else:
            sample["change_log"] = change_log
            sample["change_log"] = summerize_patient_lvl_4_summary_data(sample["change_log"], tokenizer=self.tokenizer,
                                                                        max_num_tokens=self.max_output_len,
                                                                        changelog=True)

        yaml.add_representer(float, self.float_representer, Dumper=yaml.SafeDumper)
        input = yaml.dump(sample["desc"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
        output = yaml.dump(sample["change_log"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)

        prompt = generate_and_tokenize_prompt(input, output, self.tokenizer, predict=self.predict, model_name=self.model_name,
                                              custom_attn_mask=self.custom_attn_mask, add_gen_prompt_for_predict=self.add_gen_prompt_for_predict, summary_len=self.summary_len)

        # logging.warning(f"Prompt len: {len(prompt['input_ids'])}")
        # logging.warning(f"Sample: {sample_info}")
        if self.predict == False:
            return prompt
        else:
            return {
                'input_ids': prompt['input_ids'],
                'labels': prompt['labels'],
                'attention_mask': prompt['attention_mask'],
                'stay_id': sample["stay_id"],
                'hour_idx': sample["hour_idx"],
            }


''' Dataset for training with summary embeddings: output = tokenized ed_vitals, description embeddings, summary mask -> next step '''

class SummaryDatasetEmbs(Dataset):
    def __init__(self, tokenizer, split='train', predict=False, max_input_len=1000, max_output_len=500,
                 model_name="Meta-Llama-3.1-8B-Instruct-bnb-4bit", custom_attn_mask=False, sum_type='last', merge_with_text=False, summary_len=8, used_text_hours=24, drop_augment=False, eval_mode='', summ_ratio=0.33,
                 task = '', direct_prediction_ft=False):

        self.sum_type = sum_type
        self.merge_with_text = merge_with_text
        self.summary_len = summary_len
        self.task = task
        self.direct_prediction_ft = direct_prediction_ft

        if task == 'ed_admission': # fine-tuning for ED Admission Prediction
            if direct_prediction_ft: # instead of fine-tuning the full pathway prediction, fine-tune for directly predicting the result in the next step without simulation
                train_data_path = f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz"
                val_data_path = f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz"
            else:
                train_data_path = f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_ed_admission_all.json.gz"
                val_data_path = f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_admission_all.json.gz"

        elif task == 'ed_icd':
            train_data_path = f"{mimiciv_path}finetuning_data/train_noweighting_sample_timepoint_paths_ed_icd_all.json.gz"
            val_data_path = f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_icd_all.json.gz"

        elif task == 'mort':
            train_data_path = f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_icu_mort.json.gz"
            val_data_path = f"{mimiciv_path}finetuning_data/val_weighting_sample_timepoint_paths_icu_mort.json.gz"

        elif task == 'los':
            train_data_path = f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_icu_los_path.json.gz"
            val_data_path = f"{mimiciv_path}finetuning_data/{split}_weighting_sample_timepoint_paths_icu_los_path.json.gz"

        elif task == 'hosp_icd':
            train_data_path = f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_hosp_icd_all.json.gz"
            val_data_path = f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_hosp_icd_all.json.gz"

        else:
            train_data_path = f"{mimiciv_path}{split}_weighting_sample_timepoint_paths_1Mio.json.gz"
            val_data_path = f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz"

        if split == 'train':
            with gzip.open(train_data_path, "rt") as f:
                self.data = json.load(f)
        else:
            with gzip.open(val_data_path, "rt") as f:
                self.data = json.load(f)
            self.data = self.data[:5000]

        self.data = sorted(self.data) #idx of get_item is NOT in order so this is fine - just so that val samples are always the same

        if predict:  # for generation and metric calculation, we only need a subset of the data, as it takes too long otherwise
            # select 200 random, but fixed samples for qualitative evaluation using sample
            # random.seed(42)
            # self.data = random.sample(self.data, min(len(self.data), 200))
            self.data = self.data[:5000] #500 was for during training, 5000 for evaluation
            print(f"Predicting on {len(self.data)} samples.")
        self.tokenizer = tokenizer
        self.split = split
        self.predict = predict

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.model_name = model_name
        self.custom_attn_mask = custom_attn_mask
        self.used_text_hours = used_text_hours
        self.drop_augment = drop_augment
        self.eval_mode = eval_mode
        self.summ_ratio = summ_ratio

    def __len__(self):
        return len(self.data)

    def float_representer(self, dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def __getitem__(self, idx):

        if self.task == 'ed_admission':
            if self.direct_prediction_ft:
                lvl1_text_path = "all_data_ed_admission_pred_inital_timepoint_nohourrestriction/"
            else:
                lvl1_text_path = "all_data_ed_admission_pred_all_nohourrestriction/"

        elif self.task == 'ed_icd':
            lvl1_text_path = "all_data_ed_icd_pred_all_nohourrestriction/"

        else:
            lvl1_text_path = "all_data_summarylvl1_los_noisy/"

        if os.path.exists(mimiciv_path + lvl1_text_path):
            data_path = mimiciv_path
        else:
            data_path = mimiciv_path
        sample_info = self.data[idx]
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        stay_id = sample_path[5:-5]
        sample = json.load(open(data_path + lvl1_text_path + sample_path, "r"))
        sample = sample[timepoint_idx]
        yaml.add_representer(float, self.float_representer, Dumper=yaml.SafeDumper)

        if self.task == 'ed_admission' or self.task == 'ed_icd': # accidentally used the same path for both tasks to save the summaries
            summary_path = "all_summaries_last_int_ed_stays/"
        else:
            summary_path = "all_summaries_last_int/"

        emb_path = data_path + summary_path + stay_id + "_" + timepoint_idx + ".pt.gz"
        with gzip.open(emb_path, 'rb') as f:
            desc_emb = torch.load(f)
            # cast to bfloat16
            desc_emb = {key: value.type(torch.bfloat16) for key, value in desc_emb.items()}

        # load text information if merged version is used
        if self.merge_with_text: # summary and text
            # dataset_name == "24h" or '12h' or '1h'
            if self.task == 'ed_admission':
                if self.direct_prediction_ft:
                    text_data_path = "all_data_ed_admission_pred_inital_timepoint/"
                else:
                    text_data_path = "all_data_ed_admission_pred_all/"
            elif self.task == 'ed_icd':
                text_data_path = "all_data_ed_icd_pred_all/"
            elif self.task == 'mort':
                if self.direct_prediction_ft:
                    text_data_path = "all_data_icu_mort_pred_inital_timepoint/"
                else:
                    text_data_path = "all_data_icu_mort_pred_all/"
            elif self.task == 'los':
                if self.direct_prediction_ft:
                    text_data_path = "all_data_icu_los_alltimepoints_pred_inital_timepoint/"
                else:
                    text_data_path = "all_data_icu_los_pred_all/"
            elif self.task == 'hosp_icd':
                text_data_path = "all_data_hosp_icd_pred_all/"
            else:
                text_data_path = f"all_data_{self.used_text_hours}_hours_los_noisy/"
            try:
                sample_textual = json.load(open(data_path + text_data_path + sample_path, "r"))
            except Exception as e:
                print(f"Error loading sample {sample_path}: {e}", flush=True)
                raise e
            sample_textual = sample_textual[timepoint_idx]

            # dynamically adjust input and output length
            SUMMARY_LENGTH = self.summary_len
            sample_textual["desc"] = summerize_patient_lvl_4(sample_textual["desc"], tokenizer=self.tokenizer, max_num_tokens=self.max_input_len-len(desc_emb)*SUMMARY_LENGTH)
            sample_textual["change_log"] = summerize_patient_lvl_4(sample_textual["change_log"], tokenizer=self.tokenizer, max_num_tokens=self.max_output_len, changelog=True)

            input = yaml.dump(sample_textual["desc"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
            output = yaml.dump(sample_textual["change_log"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)

        else: # only summary
            # los_only samples
            sample_textual = json.load(open(data_path + f"all_data_losonly_los_noisy/" + sample_path, "r"))
            sample_textual = sample_textual[timepoint_idx]

            input = yaml.dump(sample_textual["desc"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
            output = yaml.dump(sample_textual["change_log"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)

        prompt, sum_mask = generate_and_tokenize_prompt_full_summary(input, output, self.tokenizer, predict=self.predict, model_name=self.model_name,
                                                                     sum_emb_dict=desc_emb, summary_len=self.summary_len, drop_augment = self.drop_augment,
                                                                     eval_mode=self.eval_mode, summ_ratio=self.summ_ratio)
        # flatten desc_emb in order they have to be placed in the prompt
        # if sum_mask is empty because summary was dropped, create empty tensor
        desc_emb = torch.stack([value for key, value in desc_emb.items()], dim=0) if sum(sum_mask) > 0 else torch.tensor([], dtype=torch.bfloat16)

        if self.predict == False:
            prompt["desc_emb"] = desc_emb
            prompt["sum_mask"] = sum_mask
            return prompt
        else:
            return {
                'input_ids': prompt['input_ids'],
                'labels': prompt['labels'],
                'attention_mask': prompt['attention_mask'],
                'stay_id': sample["stay_id"],
                'hour_idx': sample["hour_idx"],
                'desc_emb': desc_emb,
                'sum_mask': sum_mask
            }


''' Dataset for precomputing summary embs for all sections: iterates over patient sections, splits them if too long, returns one section at a time in order '''
class SectionIterableDataset(IterableDataset):
    def __init__(self, tokenizer, split='train', predict=False, max_input_len=4000, max_output_len=1000,
                 model_name="Meta-Llama-3.1-8B-Instruct-bnb-4bit", weighted_sampling=False, custom_attn_mask=False,
                 dataset_name="24h_los", add_gen_prompt_for_predict=True, summary_len=8, dataset_idx = 0, task=""):
        """
        Initializes the iterable dataset by setting up data sources and configurations.

        Args:
            tokenizer: Tokenizer to be used for processing text.
            split (str): Dataset split ('train', 'validation', 'test').
            predict (bool): Flag indicating whether the dataset is for prediction.
            max_input_len (int): Maximum length for input sequences.
            max_output_len (int): Maximum length for output sequences.
            model_name (str): Name of the model being used.
            weighted_sampling (bool or str): Determines if weighted sampling is used.
            custom_attn_mask (bool): Flag for using custom attention masks.
            dataset_name (str): Name of the dataset variant.
            predict_all (bool): Flag to determine if all samples should be used for prediction.
            add_gen_prompt_for_predict (bool): Flag to add generation prompts during prediction.
            mimiciv_path (str): Path to the MIMIC-IV data.
        """
        super(SectionIterableDataset, self).__init__()
        self.add_gen_prompt_for_predict = add_gen_prompt_for_predict
        self.split = split
        self.predict = predict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.model_name = model_name
        self.custom_attn_mask = custom_attn_mask
        self.dataset_name = dataset_name
        self.summary_len = summary_len
        self.task = task

        suffix = ""
        if task == "ed_admission":
            suffix = "_ed_stays"
        elif task == "ed_icd":
            suffix = "_ed_stays_icd"
        elif task == "los":
            suffix = "_icu_los"

        with open(mimiciv_path + f"sample_to_sections_{split}{suffix}.json", "r") as f:
            self.sample_to_section_dict = json.load(f)

        if dataset_idx == 0:
            # only first 25%
            self.sample_to_section_dict = dict(list(self.sample_to_section_dict.items())[:int(len(self.sample_to_section_dict)/4)])
        elif dataset_idx == 1:
            # only second 25%
            self.sample_to_section_dict = dict(list(self.sample_to_section_dict.items())[int(len(self.sample_to_section_dict)/4):int(len(self.sample_to_section_dict)/2)])
        elif dataset_idx == 2:
            # only third 25%
            self.sample_to_section_dict = dict(list(self.sample_to_section_dict.items())[int(len(self.sample_to_section_dict)/2):int(3*len(self.sample_to_section_dict)/4)])
        elif dataset_idx == 3:
            # only last 25%
            self.sample_to_section_dict = dict(list(self.sample_to_section_dict.items())[int(3*len(self.sample_to_section_dict)/4):])
        else: # no split
            pass

        # drop samples already processed:
        processed_files = os.listdir(mimiciv_path + f"all_summaries_last_int{suffix}/")
        processed_ids = [file_name[:-6] for file_name in processed_files]
        self.sample_to_section_dict = {key: value for key, value in self.sample_to_section_dict.items() if key[5:].replace('.json', '') not in processed_ids}

        print(f"Loaded {len(self.sample_to_section_dict)} samples for '{split}' split for dataset idx {dataset_idx}.")

        # create data including all (sample, section) pairs from sample_to_section_dict
        self.data_sections = [
            (sample, section)
            for sample, sections in self.sample_to_section_dict.items()
            for section in sections
        ]

        self.patients = list(self.sample_to_section_dict.keys())

        print(f"Loaded {len(self.data_sections)} sections for '{split}' split for dataset idx {dataset_idx}.")

    def float_representer(self, dumper, value):
        """
        Custom YAML float representer to ensure floats are represented as strings.
        """
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))

    def process_patient_data(self, args):
        """Process a single patient's data and return collected samples."""
        sample_info, block_idx, idx, mimiciv_path, sections, self_attrs = args
        dataset_name = self_attrs['dataset_name']
        tokenizer = self_attrs['tokenizer']
        max_input_len = self_attrs['max_input_len']
        predict = self_attrs['predict']
        model_name = self_attrs['model_name']
        custom_attn_mask = self_attrs['custom_attn_mask']
        add_gen_prompt_for_predict = self_attrs['add_gen_prompt_for_predict']
        summary_len = self_attrs['summary_len']
        sample_to_section_dict = self_attrs['sample_to_section_dict']

        collected_samples = []
        for section_stay in sample_to_section_dict[sample_info]:
            sample_path, timepoint_idx = sample_info.rsplit("_", 1)
            section, stay = section_stay

            if self.task == 'ed_admission':
                if timepoint_idx == '0':
                    file_path = mimiciv_path + "all_data_ed_admission_pred_inital_timepoint_nohourrestriction/" + sample_path
                else:
                    file_path = mimiciv_path + "all_data_ed_admission_pred_all_nohourrestriction/" + sample_path
                if not os.path.exists(file_path):
                    print( f"File {file_path} with timepoint {timepoint_idx} does not exist")
                    file_path = mimiciv_path + "all_data_ed_admission_pred_all_nohourrestriction/" + sample_path

            elif self.task == 'ed_icd':
                file_path = mimiciv_path + "all_data_ed_icd_pred_all_nohourrestriction/" + sample_path

            elif self.task == 'los':
                file_path = mimiciv_path + "all_data_icu_los_pred_inital_timepoint_nohourrestriction/" + sample_path

            else:
                if dataset_name == "24h":
                    file_path = mimiciv_path + "all_data_24_hours/" + sample_path
                elif dataset_name == "all_steps":
                    file_path = mimiciv_path + "all_data_summarylvl1_los_noisy/" + sample_path
                else:
                    raise ValueError("Invalid dataset_name")

            with open(file_path, "r") as f:
                sample = json.load(f)

            sample = sample[timepoint_idx]
            section_group = sections[section]
            is_icu = section_group[0][0] == "ICU"

            if is_icu:
                if len(section_group[0]) == 2:
                    desc, change_log = get_section_info(sample, sample_info, section,
                                                        section_key="ICU Stay",
                                                        sub_keys=[section[1] for section in section_group],
                                                        stay=stay)
                elif len(section_group[0]) == 3:
                    desc, change_log = get_section_info(sample, sample_info, section,
                                                        section_key="ICU Stay",
                                                        sub_keys=[section_group[0][1]],
                                                        sub_sub_keys=[section[2] for section in section_group],
                                                        stay=stay)
                else:
                    raise ValueError("Invalid section group")
            else:
                if len(section_group[0]) == 1:
                    desc, change_log = get_section_info(sample, sample_info, section, section_key=section_group[0][0])
                elif len(section_group[0]) == 2:
                    desc, change_log = get_section_info(sample, sample_info, section,
                                                        section_key=section_group[0][0],
                                                        sub_keys=[section[1] for section in section_group])
                else:
                    raise ValueError("Invalid section group")

            if desc is None:
                continue

            parts = split_section_into_parts(desc, tokenizer=tokenizer, max_num_tokens=max_input_len, changelog=False)
            if len(parts) > 4:
                # randomly select 4 parts: this happens in less than 0.01% of the cases if parting was accidentally too sub-optimal -> dropping columns only happens VERY rarely
                parts = random.sample(parts, 4)
            yaml.add_representer(float, self_attrs['float_representer'], Dumper=yaml.SafeDumper)

            for part_idx, part in enumerate(parts):
                input = yaml.dump(part, sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
                prompt = generate_and_tokenize_prompt(input, "", tokenizer, predict=predict,
                                                      model_name=model_name, custom_attn_mask=custom_attn_mask,
                                                      gen_summ_mask=False, add_gen_prompt_for_predict=add_gen_prompt_for_predict,
                                                      summary_len=summary_len)
                collected_samples.append((len(prompt['input_ids']), {
                    'input_ids': prompt['input_ids'],
                    'labels': prompt['labels'],
                    'attention_mask': prompt['attention_mask'],
                    'stay_id': (sample.get('stay_id', -1), stay, section, part_idx),
                    'hour_idx': int(timepoint_idx),  # Defaulting to -1 if not present
                    'block_idx': block_idx
                }))
        return collected_samples

    def collect_samples_threaded(self, block, block_idx, mimiciv_path, sections, self_attrs, num_threads=8):
        """Collect samples using threads."""
        args = [
            (sample_info, block_idx, idx, mimiciv_path, sections, self_attrs)
            for idx, sample_info in enumerate(block)
        ]
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.process_patient_data, args))

        # Flatten the results
        return [sample for patient_samples in results for sample in patient_samples]

    def __iter__(self):
        """
        Iterates over the dataset, yielding processed samples.

        Yields:
            Depending on the 'predict' flag:
                - For training: prompt
                - For prediction: dict with input_ids, labels, attention_mask, stay_id, hour_idx
        """

        worker_id = 0
        num_workers = 1

        # Partition the data among workers
        total_samples = len(self.patients)
        samples_per_worker = math.ceil(total_samples / num_workers)
        start = worker_id * samples_per_worker
        end = min(start + samples_per_worker, total_samples)
        assigned_data = self.patients[start:end]

        print(f"Worker {worker_id} processing samples {start} to {end}.")

        # split assigned_data in blocks of 100 patients
        assigned_data_blocks = [assigned_data[i:i + 100] for i in range(0, len(assigned_data), 100)]

        for block_idx, block in enumerate(assigned_data_blocks):
            print("Processing block", block_idx)
            start_time = time.time()
            self_attrs = {
                'dataset_name': self.dataset_name,
                'tokenizer': self.tokenizer,
                'max_input_len': self.max_input_len,
                'predict': self.predict,
                'model_name': self.model_name,
                'custom_attn_mask': self.custom_attn_mask,
                'add_gen_prompt_for_predict': self.add_gen_prompt_for_predict,
                'summary_len': self.summary_len,
                'float_representer': self.float_representer,
                'sample_to_section_dict': self.sample_to_section_dict
            }

            collected_samples = self.collect_samples_threaded(block, block_idx, mimiciv_path, sections, self_attrs)

            # sort collected samples by length
            collected_samples = sorted(collected_samples, key=lambda x: x[0], reverse=True)
            print("Finished processing block", block_idx)
            print("Time taken:", time.time() - start_time)
            for sample in collected_samples:
                yield sample[1]

def update_section_dict(sample, section_dict, sample_info, section_name, section_group, section_key, sub_keys=None, sub_sub_keys=None, stay=""):
    global processed_patients
    static_sections = ['HS_General__HA_PatientLocation__HA_CareTaker', 'HS_OutpatientMeasurements', 'HS_RadiologyNotes', 'ICU_CE_MDProgressNote',
                       'ICU_CE_AdmHistory']

    # Handle ICU stays if a specific stay is provided
    if stay != "":
        desc = sample.get("desc", {}).get(section_key, {}).get(stay, {})
    else:
        # For non-ICU sections or where stay is not relevant
        desc = sample.get("desc", {}).get(section_key, {})

    # Access deeper levels if needed
    if sub_keys:
        # merge sections of all sub_keys
        desc = {sub_key: desc.get(sub_key, {}) for sub_key in sub_keys if desc.get(sub_key, {}) != {}}
    if sub_sub_keys:
        # merge sections of all sub_sub_keys
        desc = {sub_sub_key: desc.get(sub_keys[0], {}).get(sub_sub_key, {}) for sub_sub_key in sub_sub_keys if
                desc.get(sub_keys[0], {}).get(sub_sub_key, {}) != {}}

    # Determine if there is any content in desc
    if desc != {}:
        if section_name in static_sections:
            if sample_info.rsplit("_", 1)[0] not in processed_patients[section_name]:
                section_dict[section_name].append((sample_info, stay))
                processed_patients[section_name].append(sample_info.rsplit("_", 1)[0])
            return section_dict

        # Determine if section has changed or is empty in the change log
        if stay != "":
            change_log = sample.get("change_log", {}).get(section_key, {}).get(stay, {})
        else:
            change_log = sample.get("change_log", {}).get(section_key, {})

        # Access deeper levels if needed
        if change_log != {}:
            if sub_keys:
                change_log = {sub_key: change_log.get(sub_key, {}) for sub_key in sub_keys if change_log.get(sub_key, {}) != {}}
            if sub_sub_keys:
                change_log = {sub_sub_key: change_log.get(sub_keys[0], {}).get(sub_sub_key, {}) for sub_sub_key in sub_sub_keys if
                              change_log.get(sub_keys[0], {}).get(sub_sub_key, {}) != {}}

        if change_log != {}:
            section_dict[section_name].append((sample_info, stay))
            return section_dict
        else:
            section_dict[f"{section_name}-empty"].append((sample_info, stay))
            return section_dict

    return section_dict

def check_sample(arguments):
    sample_info, mimiciv_path, sections = arguments
    sample_path, timepoint_idx = sample_info.rsplit("_", 1)

    # Load sample
    sample = json.load(open(f"{mimiciv_path}all_data_summarylvl1_los_noisy/{sample_path}", "r"))
    sample = sample[timepoint_idx]

    section_dict = defaultdict(list)

    # Iterate over each section group and process
    for section_name, section_group in sections.items():
        is_icu = section_group[0][0] == "ICU"

        if is_icu:
            if "ICU Stay" in sample["desc"]:
                for stay in sample["desc"]["ICU Stay"]:
                    if len(section_group[0]) == 2:
                        section_dict = update_section_dict(sample, section_dict, sample_info, section_name, section_group,
                                                           section_key="ICU Stay", sub_keys=[section[1] for section in section_group], stay=stay)
                    elif len(section_group[0]) == 3:
                        section_dict = update_section_dict(sample, section_dict, sample_info, section_name, section_group,
                                                           section_key="ICU Stay", sub_keys=[section_group[0][1]],
                                                           sub_sub_keys=[section[2] for section in section_group], stay=stay)

        else:
            if len(section_group[0]) == 1:
                section_dict = update_section_dict(sample, section_dict, sample_info, section_name, section_group, section_key=section_group[0][0])

            elif len(section_group[0]) == 2:
                section_dict = update_section_dict(sample, section_dict, sample_info, section_name, section_group, section_key=section_group[0][0],
                                                   sub_keys=[section[1] for section in section_group])

    return section_dict

def get_valid_indices(data, mimiciv_path, sections, num_workers=200):
    # Prepare arguments for multiprocessing
    args = [(sample_info, mimiciv_path, sections) for sample_info in data]

    # Initialize the Pool with the desired number of workers
    with Pool(processes=num_workers) as pool:
        # Use `imap_unordered` to get results as they complete and feed the data
        results = pool.imap_unordered(check_sample, args)

        # Track progress with tqdm
        section_dicts = []
        with tqdm(total=len(data)) as pbar:
            for result in results:
                section_dicts.append(result)
                pbar.update(1)

    final_section_dict = defaultdict(list)
    for section_dict in section_dicts:
        for section_group, sample_infos in section_dict.items():
            final_section_dict[section_group].extend(sample_infos)

    return final_section_dict

def process_patient_data_sample(sample_sections, sample, tokenizer, float_representer, model_name, timepoint_idx):
    """Process a single patient's data and return collected samples."""

    sample_info = "" #only used for logging

    collected_samples = []

    for section_stay in sample_sections:
        section, stay = section_stay

        section_group = sections[section]
        is_icu = section_group[0][0] == "ICU"

        if is_icu:
            if len(section_group[0]) == 2:
                desc, change_log = get_section_info(sample, sample_info, section,
                                                    section_key="ICU Stay",
                                                    sub_keys=[section[1] for section in section_group],
                                                    stay=stay)
            elif len(section_group[0]) == 3:
                desc, change_log = get_section_info(sample, sample_info, section,
                                                    section_key="ICU Stay",
                                                    sub_keys=[section_group[0][1]],
                                                    sub_sub_keys=[section[2] for section in section_group],
                                                    stay=stay)
            else:
                raise ValueError("Invalid section group")
        else:
            if len(section_group[0]) == 1:
                desc, change_log = get_section_info(sample, sample_info, section, section_key=section_group[0][0])
            elif len(section_group[0]) == 2:
                desc, change_log = get_section_info(sample, sample_info, section,
                                                    section_key=section_group[0][0],
                                                    sub_keys=[section[1] for section in section_group])
            else:
                raise ValueError("Invalid section group")

        if desc is None:
            continue

        parts = split_section_into_parts(desc, tokenizer=tokenizer, max_num_tokens=5000, changelog=False)
        if len(parts) > 4:
            # randomly select 4 parts: this happens in less than 0.01% of the cases if parting was accidentally too sub-optimal -> dropping columns only happens VERY rarely
            parts = random.sample(parts, 4)
        yaml.add_representer(float, float_representer, Dumper=yaml.SafeDumper)

        for part_idx, part in enumerate(parts):
            input = yaml.dump(part, sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
            prompt = generate_and_tokenize_prompt(input, "", tokenizer, predict=True,
                                                  model_name=model_name, custom_attn_mask=True,
                                                  gen_summ_mask=False, add_gen_prompt_for_predict=False,
                                                  summary_len=8)

            collected_samples.append((len(prompt['input_ids']), {
                'input_ids': prompt['input_ids'],
                'labels': prompt['labels'],
                'attention_mask': prompt['attention_mask'],
                'stay_id': (sample.get('stay_id', -1), stay, section, part_idx),
                'hour_idx': int(timepoint_idx)
            }))

    return collected_samples

def get_sample_sections(sample_info, sections, tokenizer, float_representer, model_name, timepoint_idx, sample=None):

    # Load sample
    if sample is None:
        sample_path, timepoint_idx = sample_info.rsplit("_", 1)
        sample = json.load(open(f"{mimiciv_path}all_data_summarylvl1_los_noisy/{sample_path}", "r"))
        sample = sample[timepoint_idx]

    sample_sections = []

    for section in sections:
        section_group = sections[section]
        is_icu = section_group[0][0] == "ICU"
        if is_icu:
            if "ICU Stay" in sample["desc"]:
                for stay in sample["desc"]["ICU Stay"]:
                    if len(section_group[0]) == 2:
                        if section_group[0][1] in sample["desc"]["ICU Stay"][stay]:
                            sample_sections.append((section, stay))
                    elif len(section_group[0]) == 3:
                        if section_group[0][1] in sample["desc"]["ICU Stay"][stay]:
                            if section_group[0][2] in sample["desc"]["ICU Stay"][stay][section_group[0][1]]:
                                sample_sections.append((section, stay))

        else:
            if section_group[0][0] in sample["desc"]:
                if len(section_group[0]) == 1:
                    sample_sections.append((section, ""))
                elif len(section_group[0]) == 2:
                    for elem_idx, section_elem in enumerate(section_group):
                        if section_elem[1] in sample["desc"][section_group[elem_idx][0]]:
                            sample_sections.append((section, ""))
                            break

    collected_embs = process_patient_data_sample(sample_sections, sample, tokenizer, float_representer, model_name, timepoint_idx)

    return collected_embs

def get_sample_sections_extract(sample_info, mimiciv_path, sections, task=''):
    sample_path, timepoint_idx = sample_info.rsplit("_", 1)

    # Load sample
    if task == 'los':
        sample = json.load(open(f"{mimiciv_path}all_data_icu_los_pred_inital_timepoint_nohourrestriction/{sample_path}", "r"))
    else:
        sample = json.load(open(f"{mimiciv_path}all_data_summarylvl1_los_noisy/{sample_path}", "r"))
    sample = sample[timepoint_idx]
    sample_sections = []

    for section in sections:
        section_group = sections[section]
        is_icu = section_group[0][0] == "ICU"
        if is_icu:
            if "ICU Stay" in sample["desc"]:
                for stay in sample["desc"]["ICU Stay"]:
                    if len(section_group[0]) == 2:
                        if section_group[0][1] in sample["desc"]["ICU Stay"][stay]:
                            sample_sections.append((section, stay))
                    elif len(section_group[0]) == 3:
                        if section_group[0][1] in sample["desc"]["ICU Stay"][stay]:
                            if section_group[0][2] in sample["desc"]["ICU Stay"][stay][section_group[0][1]]:
                                sample_sections.append((section, stay))

        else:
            if section_group[0][0] in sample["desc"]:
                if len(section_group[0]) == 1:
                    sample_sections.append((section, ""))
                elif len(section_group[0]) == 2:
                    for elem_idx, section_elem in enumerate(section_group):
                        if section_elem[1] in sample["desc"][section_group[elem_idx][0]]:
                            sample_sections.append((section, ""))
                            break

    return sample_sections

def create_section_to_sample_dict(sections, split="train"):
    # Adjust paths as per your setup
    if split == "train":
        with gzip.open(f"{mimiciv_path}train_weighting_sample_timepoint_paths_1Mio.json.gz", "rt") as f:
            data = json.load(f)
    else:
        with gzip.open(f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz", "rt") as f:
            data = json.load(f)

    final_section_dict = get_valid_indices(data, mimiciv_path, sections)

    # Save valid_idxs
    with open(mimiciv_path + f"samples_with_sections_{split}.json", "w") as f:
        json.dump(final_section_dict, f)


def sample_summary_dataset(split="train"):
    with open(mimiciv_path + f"samples_with_sections_{split}.json", "r") as f:
        section_dict = json.load(f)

    # drop sections ending with -empty
    section_dict_clean = {section: samples for section, samples in section_dict.items() if not section.endswith("-empty")}

    # sample 500.000 samples using log weighting of amounts in section_dict

    # get the amount of samples in each section
    section_amounts = {section: len(samples) for section, samples in section_dict_clean.items()}

    # get the weights for each section
    section_weights = {section: np.log(amount) for section, amount in section_amounts.items()}

    # Normalize the weights to sum to 1 to get probabilities
    total_weight = sum(section_weights.values())
    section_props = {section: weight / total_weight for section, weight in section_weights.items()}

    # sample 1000.000 samples
    sample_amount = 1000000 if split == "train" else 2000
    sampled_sections_counter = {section: 0 for section in section_dict_clean.keys()}
    sampled_samples = []
    for i in range(sample_amount):
        sampled_section = np.random.choice(list(section_dict_clean.keys()), p=list(section_props.values()))
        sampled_sections_counter[sampled_section] += 1
        sampled_samples.append((section_dict_clean[sampled_section][np.random.choice(len(section_dict_clean[sampled_section]))], sampled_section))

    # for each section, sample 10% of amount of samples in that section from "empty" sections if available
    for section, samples in section_dict.items():
        if section.endswith("-empty"):
            sampled_empty_samples = section_dict[section]
            sampled_empty_samples = random.sample(sampled_empty_samples, int(0.1 * sampled_sections_counter[section[:-6]]))
            sampled_samples.extend([(sample, section[:-6]) for sample in sampled_empty_samples])

    # save sampled samples
    with open(mimiciv_path + f"section_samples_for_summary_{split}.json", "w") as f:
        json.dump(sampled_samples, f)


class CustomDataCollatorForSeq2Seq:
    def __init__(self, base_collator, additional_keys=[]):
        self.base_collator = base_collator
        self.additional_keys = additional_keys

    def __call__(self, features):
        # Separate standard model inputs and additional fields
        standard_features = [{k: v for k, v in f.items() if k not in self.additional_keys} for f in features]
        additional_features = {key: [f[key] for f in features] for key in self.additional_keys}

        # Use the base collator for standard inputs
        collated_features = self.base_collator(standard_features)

        if 'sum_mask' in additional_features:
            max_length = max(len(f) for f in additional_features["sum_mask"])
            max_length = ((max_length + 8 - 1) // 8) * 8

            if self.base_collator.tokenizer.padding_side == 'right':
                padded_features = [
                    F.pad(torch.tensor(f, dtype=torch.bool), (0, max_length - len(f)), value=False)
                    for f in additional_features['sum_mask']
                ]
            else: # padding_side == 'left'
                padded_features = [
                    F.pad(torch.tensor(f, dtype=torch.bool), (max_length - len(f), 0), value=False)
                    for f in additional_features['sum_mask']
                ]
            collated_features['sum_mask'] = torch.stack(padded_features)
            additional_features.pop('sum_mask')

        if 'desc_emb' in additional_features:
            collated_features['desc_emb'] = torch.cat(additional_features['desc_emb'])
            additional_features.pop('desc_emb')

        # Add back the additional fields without modifying them
        collated_features.update(additional_features)

        return collated_features


def process_sample(args):
    sample_info, mimiciv_path, sections = args
    sample_sections = get_sample_sections_extract(sample_info, mimiciv_path, sections)
    return sample_info, sample_sections

def create_sample_to_section_dict(split, sections, num_workers):
    if split == "train":
        with gzip.open(f"{mimiciv_path}train_weighting_sample_timepoint_paths_1Mio.json.gz", "rt") as f:
            data = json.load(f)
    else:
        with gzip.open(f"{mimiciv_path}{split}_noweighting_sample_timepoint_paths.json.gz", "rt") as f:
            data = json.load(f)

    # Prepare arguments for the pool
    args = [(sample_info, mimiciv_path, sections) for sample_info in data]

    # Use Pool to parallelize the loop
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_sample, args)

    # Collect results into a dictionary
    sample_to_sections_dict = {sample_info: sample_sections for sample_info, sample_sections in results}

    # Save valid_idxs
    with open(mimiciv_path + f"sample_to_sections_{split}.json", "w") as f:
        json.dump(sample_to_sections_dict, f)

def create_sample_to_section_dict_ed_adm(split, task='ed_admission'):

    if task == 'ed_admission':
        if split == "train":
            with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz", "rt") as f:
                data = json.load(f)
            with gzip.open(f"{mimiciv_path}finetuning_data/train_weighting_sample_timepoint_paths_ed_admission_all.json.gz", "rt") as f:
                data_all = json.load(f)
            data = data + data_all

        else:
            with gzip.open(f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_admission_inital_timepoint.json.gz", "rt") as f:
                data = json.load(f)
            with gzip.open(f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_admission_all.json.gz", "rt") as f:
                data_all = json.load(f)
            data = data + data_all

    elif task == 'ed_icd':
        if split == "train":
            with gzip.open(f"{mimiciv_path}finetuning_data/train_noweighting_sample_timepoint_paths_ed_icd_all.json.gz", "rt") as f:
                data = json.load(f)

        else:
            with gzip.open(f"{mimiciv_path}finetuning_data/{split}_noweighting_sample_timepoint_paths_ed_icd_all.json.gz","rt") as f:
                data = json.load(f)

    # For ED_Stay it's always the same section
    results = [(sample_info, [('ED_Stay', '')]) for sample_info in data]

    # Collect results into a dictionary
    sample_to_sections_dict = {sample_info: sample_sections for sample_info, sample_sections in results}

    # Save valid_idxs
    with open(mimiciv_path + f"sample_to_sections_{split}_ed_stays_icd.json", "w") as f:
        json.dump(sample_to_sections_dict, f)


if __name__ == '__main__':
    ''' Warning: these methods are quite I/O intensive, so it is recommended to run this on a machine with SSDs and enough RAM. '''

    sections = {'ED_Stay': [('Emergency Department Stay',)],
                'HS_General__HA_PatientLocation__HA_CareTaker': [('Hospital Stay', 'General'), ('Hospital Admission', 'Patient Location'),
                                                                 ('Hospital Admission', 'Care Taker')],
                'HS_OutpatientMeasurements': [('Hospital Stay', 'Outpatient Measurements')],
                'HS_LabResults': [('Hospital Stay', 'Lab Results')],
                'HS_RadiologyNotes': [('Hospital Stay', 'Radiology Notes')],
                'HS_Prescriptions': [('Hospital Stay', 'Prescriptions')],
                'HS_Procedures': [('Hospital Stay', 'Procedures')],
                'HS_Microbiology': [('Hospital Stay', 'Microbiology Growth Results')],
                'ICU_Medication': [('ICU', 'Medication')],
                'ICU_Output': [('ICU', 'Output')],
                'ICU_Procedures': [('ICU', 'Procedures')],
                'ICU_CE_VitalSigns': [('ICU', 'Chart Events', 'RoutineVitalSigns')],
                'ICU_CE_AdmHistory': [('ICU', 'Chart Events', 'AdmHistory_FHPA')],
                'ICU_CE_MDProgressNote': [('ICU', 'Chart Events', 'MDProgressNote')],
                'ICU_CE_Respiratory': [('ICU', 'Chart Events', 'Respiratory')],
                'ICU_CE_Pulmonary': [('ICU', 'Chart Events', 'Pulmonary')],
                'ICU_CE_SkinAssessment': [('ICU', 'Chart Events', 'Skin-Assessment')],
                'ICU_CE_SkinImpairment': [('ICU', 'Chart Events', 'Skin-Impairment')],
                'ICU_CE_SkinIncisions': [('ICU', 'Chart Events', 'Skin-Incisions')],
                'ICU_CE_CardioPulses': [('ICU', 'Chart Events', 'Cardiovascular(Pulses)')],
                'ICU_CE_Neurological': [('ICU', 'Chart Events', 'Neurological')],
                'ICU_CE_Hemodynamics': [('ICU', 'Chart Events', 'Hemodynamics')],
                'ICU_CE_Alarms': [('ICU', 'Chart Events', 'Alarms')],
                'ICU_CE_PainSedation': [('ICU', 'Chart Events', 'Pain_Sedation')],
                'ICU_CE_GIGU': [('ICU', 'Chart Events', 'GI_GU')],
                'ICU_CE_Cardio': [('ICU', 'Chart Events', 'Cardiovascular')],
                'ICU_CE_CardioPacerData': [('ICU', 'Chart Events', 'Cardiovascular(PacerData)')],
                'ICU_CE_IABP': [('ICU', 'Chart Events', 'IABP')],
                'ICU_CE_Dialysis': [('ICU', 'Chart Events', 'Dialysis')],
                'ICU_CE_Toxicology': [('ICU', 'Chart Events', 'Toxicology')],
                'ICU_CE_NICOM': [('ICU', 'Chart Events', 'NICOM')],
                }

    ''' used for generating section data to train summerizer model'''
    processed_patients = defaultdict(list)
    create_section_to_sample_dict(sections, split="train")
    create_section_to_sample_dict(sections, split="val")

    sample_summary_dataset(split="train")
    sample_summary_dataset(split="val")
