import gzip
import logging
import random
import time
from copy import deepcopy

import numpy as np
import torch
import yaml

from model_code.extract_summary_embs import extract_embs_sample
from patient_model.generate_dataset import generate_sample_description
# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from patient_model.dataset import generate_and_tokenize_prompt, generate_and_tokenize_prompt_full_summary, get_sample_sections
from patient_model.retrieve_change_log import extract_changes_from_changelog, integrate_changes_from_changelog, retrieve_change_log_for_hour
from patient_model.retrieve_patient_model import summerize_patient_lvl_4, restrict_patient_until_hour, \
    get_patient_description
from unsloth import FastLanguageModel

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

def float_representer(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:str', f"{value}")


def create_summ_embs(summ_infos, sample, tokenizer, model, model_name, ed_stay=None, summ_collator=None):
    sample_copy = deepcopy(sample)
    # move model to cuda
    model.to('cuda')
    all_extracted_embs = []
    for i in range(len(summ_infos)):
        desc = generate_sample_description(summ_infos[i]['complete_patient_stay'], summ_infos[i]['full_patient_visit'], summ_infos[i]['prior_patient_state'], sample_copy[i]['hour_idx'], summ_infos[i]['hour_time'],
                                           ed_stay=ed_stay[i])
        sample_copy[i]['desc'] = desc
        sample_sections = get_sample_sections(sample_info=None, sections=sections, sample=sample_copy[i], tokenizer=tokenizer, float_representer=float_representer, model_name=model_name, timepoint_idx=sample_copy[i]['hour_idx'])
        extracted_embs = extract_embs_sample(model, sample_sections, tokenizer, eval_data_collator=summ_collator)
        all_extracted_embs.append(extracted_embs)

    # move model back to cpu
    model.to('cpu')
    return all_extracted_embs


def generate_model_input(sample, tokenizer, model_name, batch_collator=None, use_summ=False, summ_infos=None, summ_model=None, summ_tokenizer=None, ed_stay=None, summ_collator=None, summ_only=False):
    yaml.add_representer(float, float_representer, Dumper=yaml.SafeDumper)
    yaml.add_representer(np.float64, float_representer, Dumper=yaml.SafeDumper)

    if use_summ:
        prompts = []

        desc_embs = create_summ_embs(summ_infos, sample, summ_tokenizer, summ_model, model_name, ed_stay=ed_stay, summ_collator=summ_collator)

        for sample_, desc_emb in zip(sample, desc_embs):
            desc_emb = {key: value.type(torch.bfloat16) for key, value in desc_emb.items()}

            # dynamically adjust input and output length
            SUMMARY_LENGTH = 8
            sample_["desc"] = summerize_patient_lvl_4(sample_["desc"], tokenizer=tokenizer, max_num_tokens=4000 - len(desc_emb) * SUMMARY_LENGTH)
            new_desc = {}
            if summ_only: #extract only LOS information
                if 'Emergency Department Stay' in sample_["desc"]:
                    if 'LOS' in sample_["desc"]['Emergency Department Stay']:
                        new_desc['Emergency Department Stay'] = {'LOS': sample_["desc"]['Emergency Department Stay']['LOS']}
                    else:
                        new_desc['Emergency Department Stay'] = {}

                if 'Hospital Stay' in sample_["desc"]:
                    if 'LOS' in sample_["desc"]['Hospital Stay']:
                        new_desc['Hospital Stay'] = {'LOS': sample_["desc"]['Hospital Stay']['LOS']}
                    else:
                        new_desc['Hospital Stay'] = {}

                if 'ICU Stay' in sample_["desc"]:
                    for stay in sample_["desc"]['ICU Stay']:
                        if 'LOS' in sample_["desc"]['ICU Stay'][stay]:
                            new_desc['ICU Stay'] = {stay: {'LOS': sample_["desc"]['ICU Stay'][stay]['LOS']}}
                        else:
                            new_desc['ICU Stay'] = {stay: {}}

                sample_["desc"] = new_desc

            input = yaml.dump(sample_["desc"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)

            prompt, sum_mask = generate_and_tokenize_prompt_full_summary(input, "", tokenizer, predict=True, model_name=model_name, sum_emb_dict=desc_emb, summary_len=SUMMARY_LENGTH)
            # flatten desc_emb in order they have to be placed in the prompt
            desc_emb = torch.stack([value for key, value in desc_emb.items()], dim=0)

            prompt['desc_emb'] = desc_emb
            prompt['sum_mask'] = sum_mask
            prompts.append(prompt)

        collated_prompts = batch_collator(prompts)

        return {
            'input_ids': collated_prompts['input_ids'],
            'labels': collated_prompts['labels'],
            'attention_mask': collated_prompts['attention_mask'],
            'stay_id': [sample_["stay_id"] for sample_ in sample],
            'hour_idx': [sample_["hour_idx"] for sample_ in sample],
            'desc_emb': collated_prompts['desc_emb'],
            'sum_mask': collated_prompts['sum_mask'],
        }

    else:
        # dynamically adjust input and output length
        prompts = []
        for sample_ in sample:
            sample_["desc"] = summerize_patient_lvl_4(sample_["desc"], tokenizer=tokenizer, max_num_tokens=4000)
            input = yaml.dump(sample_["desc"], sort_keys=False, default_style="'", Dumper=yaml.SafeDumper)
            output = ""
            prompt = generate_and_tokenize_prompt(input, output, tokenizer, predict=True, model_name=model_name,
                                                  custom_attn_mask=False, gen_summ_mask=False,
                                                  add_gen_prompt_for_predict=True)
            prompts.append(prompt)

        collated_prompts = batch_collator(prompts)

        return {
            'input_ids': collated_prompts['input_ids'],
            'labels': collated_prompts['labels'],
            'attention_mask': collated_prompts['attention_mask'],
            'stay_id': [sample_["stay_id"] for sample_ in sample],
            'hour_idx': [sample_["hour_idx"] for sample_ in sample],
        }


def get_sample_from_state(patient_state, current_hour_idx, ed_stay, released_from_icu, just_admitted, stay_id, add_los=False, ed_los=None, hospital_los=None, icu_los=None):
    los_dict = {'ed_los': ed_los, 'hospital_los': hospital_los, 'icu_los': icu_los}
    # actual hour_idxs start with -1, but timepoints are positions in the sample list starting at 0
    prior_patient_state, hour_time = restrict_patient_until_hour(patient_state, current_hour_idx, ed_stay,
                                                                 restrict_to_N_hours=24)  # depends on data type model was trained on
    desc = get_patient_description(patient_state, prior_patient_state, hour_time=hour_time, summary_level=1,
                                   diag_avail=(not ed_stay) or just_admitted, add_los=add_los, los_dict=los_dict)  # diag_avail describes if ED diagnosis should be included as ED stay is over

    if just_admitted:
        desc['Emergency Department Stay']['Disposition'] = 'ADMITTED'

    if released_from_icu:
        desc['Hospital Stay']['Disposition'] = 'DISCHARGED FROM ICU'

    return {'desc': desc, 'stay_id': stay_id, 'hour_idx': current_hour_idx}, hour_time


'''
icu_only: if True, only simulate ICU stay, if False, simulate whole stay
'''
def simulate_development(full_patient_visit, sample, stay_id, model, tokenizer, model_name, hour_idx, complete_patient_stay, prior_patient_state,
                         prior_time, icu_only=False, just_admitted=[], ed_stay=[], only_ed=False, max_steps=-1, batch_collator=None, use_summ=False,
                         summ_model=None, summ_tokenizer=None, summ_collator=None, summ_only=False, eval_future_ed_disp=False, eval_future_mort=False,
                         eval_future_los=False):
    model.eval()
    FastLanguageModel.for_inference(model)

    # stay_over = False
    all_stays_over = [False for _ in range(len(full_patient_visit))]
    max_steps = [max_steps for _ in range(len(full_patient_visit))]
    return_values = {}
    # while not stay_over:
    while not all(all_stays_over): # only stop if all stays are over, otherwise skip the ones that are already over
        print(f"Stay ID: {stay_id}")
        print(f"Timepoint: {hour_idx}")
        print(f"Hour Time: {prior_time}")

        # drop elems in sample that are already over
        sample = [sample[i] for i in range(len(all_stays_over)) if not all_stays_over[i]]
        summ_infos = [{'complete_patient_stay': complete_patient_stay[i], 'full_patient_visit': full_patient_visit[i], 'prior_patient_state': prior_patient_state[i], 'hour_time': prior_time[i]}  for i in range(len(all_stays_over)) if not all_stays_over[i]]

        model_input = generate_model_input(sample, tokenizer, model_name, batch_collator, use_summ, summ_infos, summ_model=summ_model, summ_tokenizer=summ_tokenizer, ed_stay=ed_stay, summ_collator=summ_collator, summ_only=summ_only)
        inputs = model_input['input_ids'].to(model.device)
        attention_mask = model_input['attention_mask'].to(model.device)
        hour_idx = model_input['hour_idx']

        if use_summ:
            generated_ids = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=4000, desc_emb=model_input['desc_emb'].to(model.device), sum_mask=model_input['sum_mask'].to(model.device))
        else:
            # generate next step
            generated_ids = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=4000)
        # decode
        generated_texts = tokenizer.batch_decode(generated_ids[:, len(inputs[0]):], skip_special_tokens=True)
        print(f"Generated Text: {generated_texts}")

        # put placeholder in generated_texts for stay_over samples
        for i, stay_over in enumerate(all_stays_over):
            if stay_over:
                generated_texts.insert(i, None)
                hour_idx.insert(i, 0)
                sample.insert(i, None)

        for idx, (generated_text, stay_over) in enumerate(zip(generated_texts, all_stays_over)):
            if not stay_over:
                (ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df,
                 procedures_icd_df, icd, disposition, inputevents, outputevents_df_all, procedures_df_all, chartevents_dict_all, disposition_icu_all,
                 ed_los, hospital_los, icu_los) = extract_changes_from_changelog(generated_text, full_patient_visit[idx], eval_future_ed_disp, eval_future_mort, eval_future_los)

                all_icu_stays = [icu for icu in disposition_icu_all.keys()] if disposition_icu_all is not None else []
                first_stay = all_icu_stays[0] if len(all_icu_stays) > 0 else None

                released_from_icu = first_stay is not None and disposition_icu_all[first_stay] is not None and disposition_icu_all[first_stay] == "DISCHARGED FROM ICU"
                entered_icu = disposition is not None and disposition == "ADMITTED TO ICU"
                just_admitted_curr = just_admitted[idx]

                if ed_stay[idx] and ed_disposition is not None:  # only relevant if we are in ED
                    if ed_disposition in ["DIED", "DISCHARGED"]:
                        stay_over = True
                        all_stays_over[idx] = True
                    elif ed_disposition == "ADMITTED":
                        just_admitted_curr = True

                if not ed_stay[idx] and disposition is not None:  # only relevant if we are in hospital
                    if disposition in ["DIED", "DISCHARGED", "ALIVE", "STAYED"]: #ALIVE and STAYED only added for fine-tuned mortality / LOS downstream task
                        stay_over = True
                        released_from_icu = True #if patient is discharged from hospital, he is also discharged from ICU (if he was in ICU)
                        all_stays_over[idx] = True
                # integrate changes in patient model, update timepoint
                new_patient_state = integrate_changes_from_changelog(generated_text, prior_patient_state=deepcopy(prior_patient_state[idx]), prior_time=prior_time[idx],
                                                                     full_patient_visit=full_patient_visit[idx], released_from_icu=released_from_icu,
                                                                     entered_icu=entered_icu)

                if new_patient_state is None:
                    raise ValueError("Error in patient model: new_patient_state was None") # None is not a valid return value anymore, instead we now return the old prior_patient_state
                    # out of format or other error
                    return_values[idx] = (None, None, None, None, None, hour_idx[idx]+1, new_patient_state)
                    all_stays_over[idx] = True

                if icu_only and released_from_icu:
                    stay_over = True
                    all_stays_over[idx] = True
                if only_ed and ed_disposition is not None:
                    stay_over = True
                    all_stays_over[idx] = True
                if stay_over:
                    # return outcome
                    return_values[idx] = (ed_disposition, ed_icd, disposition, released_from_icu, icd, hour_idx[idx]+1, new_patient_state)
                else:
                    # go to next timepoint
                    if not all_stays_over[idx]:
                        hour_idx[idx] += 1
                        sample_new, prior_time_new = get_sample_from_state(new_patient_state, current_hour_idx=hour_idx[idx], ed_stay=ed_stay[idx], released_from_icu=released_from_icu,
                                                                           just_admitted=just_admitted_curr, stay_id=stay_id[idx], ed_los=ed_los, hospital_los=hospital_los, icu_los=icu_los)
                        prior_patient_state[idx] = new_patient_state
                        sample[idx] = sample_new
                        prior_time[idx] = prior_time_new

                        if just_admitted_curr:
                            just_admitted[idx] = False
                            ed_stay[idx] = False  # patient is now admitted to Hospital

                    max_steps[idx] -= 1
                    if max_steps[idx] == 0:
                        print("Max steps reached")
                        all_stays_over[idx] = True
                        return_values[idx] = (None, None, None, False, None, hour_idx[idx]+1, new_patient_state)

    return return_values