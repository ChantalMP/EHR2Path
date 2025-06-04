import json
import re
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from tdigest import TDigest
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from mimic_iv_extraction.paths import mimiciv_path as mimiciv_path
from patient_model.retrieve_patient_model import retrieve_patient_model_


def process_patient_stay(stay_id):
    # Define the processing logic for each stay
    complete_patient_stay = stay_to_ids_dict[str(stay_id)]
    full_patient_visit, hour_time = retrieve_patient_model_(complete_stay_id=stay_id, complete_patient_stay=complete_patient_stay, hour_idx=None)
    local_min_max_dict = {}
    tdigests = defaultdict(TDigest)

    def update_tdigest(key, values):
        """ Update the t-digest with new values """
        for value in values.dropna().astype(float):
            tdigests[key].update(value)

    # Process ED vitals
    if full_patient_visit.patient_ed is not None:
        if full_patient_visit.patient_ed.ed_vital is not None:
            for vital in full_patient_visit.patient_ed.ed_vital.columns:
                if vital in ['charttime', 'subject_id', 'stay_id', 'rhythm']:
                    continue
                full_patient_visit.patient_ed.ed_vital[vital] = pd.to_numeric(full_patient_visit.patient_ed.ed_vital[vital], errors='coerce')
                if len(full_patient_visit.patient_ed.ed_vital[vital].dropna()) > 0:
                    if f"vital_{vital}" not in local_min_max_dict:
                        local_min_max_dict[f"vital_{vital}"] = [full_patient_visit.patient_ed.ed_vital[vital].dropna().min().astype(float),
                                                                full_patient_visit.patient_ed.ed_vital[vital].dropna().max().astype(float)]
                    else:
                        local_min_max_dict[f"vital_{vital}"][0] = min(local_min_max_dict[f"vital_{vital}"][0],
                                                                      full_patient_visit.patient_ed.ed_vital[vital].dropna().min().astype(float))
                        local_min_max_dict[f"vital_{vital}"][1] = max(local_min_max_dict[f"vital_{vital}"][1],
                                                                      full_patient_visit.patient_ed.ed_vital[vital].dropna().max().astype(float))

                    update_tdigest(f"vital_{vital}", full_patient_visit.patient_ed.ed_vital[vital])

    # Process lab events
    if full_patient_visit.patient_adm is not None:
        if full_patient_visit.patient_adm.labevents is not None:
            for col in full_patient_visit.patient_adm.labevents.columns[1:]:
                if not col.endswith('_valueuom') and not col.endswith('_ref_range_lower') and not col.endswith(
                        '_ref_range_upper') and not col.endswith('_flag') and not col.endswith('_priority') and not col.endswith('_comments'):
                    full_patient_visit.patient_adm.labevents[col] = pd.to_numeric(full_patient_visit.patient_adm.labevents[col], errors='coerce')
                    if len(full_patient_visit.patient_adm.labevents[col].dropna()) > 0:
                        if f"lab_{col}" not in local_min_max_dict:
                            local_min_max_dict[f"lab_{col}"] = [full_patient_visit.patient_adm.labevents[col].dropna().min().astype(float),
                                                                full_patient_visit.patient_adm.labevents[col].dropna().max().astype(float)]
                        else:
                            local_min_max_dict[f"lab_{col}"][0] = min(local_min_max_dict[f"lab_{col}"][0],
                                                                      full_patient_visit.patient_adm.labevents[col].dropna().min().astype(float))
                            local_min_max_dict[f"lab_{col}"][1] = max(local_min_max_dict[f"lab_{col}"][1],
                                                                      full_patient_visit.patient_adm.labevents[col].dropna().max().astype(float))

                        update_tdigest(f"lab_{col}", full_patient_visit.patient_adm.labevents[col])

    # Process ICU visits
    if full_patient_visit.patient_icu is not None and len(full_patient_visit.patient_icu) > 0:
        for visit in full_patient_visit.patient_icu:
            if visit.outputevents is not None:
                for col in visit.outputevents.columns[1:]:
                    if not col.endswith('_valueuom'):
                        visit.outputevents[col] = pd.to_numeric(visit.outputevents[col], errors='coerce')
                        if len(visit.outputevents[col].dropna()) > 0:
                            if f"outputs_{col}" not in local_min_max_dict:
                                local_min_max_dict[f"outputs_{col}"] = [visit.outputevents[col].dropna().min().astype(float),
                                                                        visit.outputevents[col].dropna().max().astype(float)]
                            else:
                                local_min_max_dict[f"outputs_{col}"][0] = min(local_min_max_dict[f"outputs_{col}"][0],
                                                                              visit.outputevents[col].dropna().min().astype(float))
                                local_min_max_dict[f"outputs_{col}"][1] = max(local_min_max_dict[f"outputs_{col}"][1],
                                                                              visit.outputevents[col].dropna().max().astype(float))

                            update_tdigest(f"outputs_{col}", visit.outputevents[col])

            if visit.chartevents is not None and len(visit.chartevents) > 0:
                for event_key, event_val in visit.chartevents.items():
                    if f"charts_{event_key}" not in local_min_max_dict:
                        local_min_max_dict[f"charts_{event_key}"] = {}
                    if event_val is not None:
                        for col in event_val.columns[1:]:
                            if not col.endswith('_valueuom'):
                                for idx, value in enumerate(event_val[col]):
                                    if type(value) == str:
                                        pattern_mm = r'(\d+)mm\b'
                                        pattern_cm = r'(\d+)cm\b'
                                        if re.search(pattern_mm, value):
                                            event_val[col][idx] = re.sub(pattern_mm, r'\1', value)
                                        elif re.search(pattern_cm, value):
                                            event_val[col][idx] = re.sub(pattern_cm, r'\1', value)
                                event_val[col] = pd.to_numeric(event_val[col], errors='coerce')
                                if len(event_val[col].dropna()) > 0:
                                    if col not in local_min_max_dict[f"charts_{event_key}"]:
                                        local_min_max_dict[f"charts_{event_key}"][col] = [event_val[col].dropna().min().astype(float),
                                                                                          event_val[col].dropna().max().astype(float)]
                                    else:
                                        local_min_max_dict[f"charts_{event_key}"][col][0] = min(local_min_max_dict[f"charts_{event_key}"][col][0],
                                                                                                event_val[col].dropna().min().astype(float))
                                        local_min_max_dict[f"charts_{event_key}"][col][1] = max(local_min_max_dict[f"charts_{event_key}"][col][1],
                                                                                                event_val[col].dropna().max().astype(float))

                                    update_tdigest(f"charts_{event_key}___{col}", event_val[col])

    return local_min_max_dict, tdigests


def merge_min_max_dicts(dicts):
    merged_dict = {}
    for d in dicts:
        for key, val in d.items():
            if isinstance(val, list):  # Single variable
                if key not in merged_dict:
                    merged_dict[key] = val
                else:
                    merged_dict[key][0] = min(merged_dict[key][0], val[0])
                    merged_dict[key][1] = max(merged_dict[key][1], val[1])
            else:  # Nested dict (chartevents)
                if key not in merged_dict:
                    merged_dict[key] = val
                else:
                    for sub_key, sub_val in val.items():
                        if sub_key not in merged_dict[key]:
                            merged_dict[key][sub_key] = sub_val
                        else:
                            merged_dict[key][sub_key][0] = min(merged_dict[key][sub_key][0], sub_val[0])
                            merged_dict[key][sub_key][1] = max(merged_dict[key][sub_key][1], sub_val[1])
    return merged_dict


def merge_tdigests(tdigest_list):
    """ Manually merge TDigest objects by updating a single global TDigest """
    merged_tdigests = {}

    for tdigests in tdigest_list:
        for key, tdigest in tdigests.items():
            if key not in merged_tdigests:
                merged_tdigests[key] = tdigest

            else:
                merged_tdigests[key] = merged_tdigests[key] + tdigest

    return merged_tdigests


if __name__ == '__main__':
    ''' Warning: these methods are quite I/O intensive, so it is recommended to run this on a machine with SSDs and enough RAM. '''

    with open(f"{mimiciv_path}stay_to_ids_dict.json", "r") as f:
        stay_to_ids_dict = json.load(f)

    stay_to_ids_dict = {k: v for k, v in stay_to_ids_dict.items()}
    # stay_to_ids_dict = {k: v for k, v in list(stay_to_ids_dict.items())[:100]}

    with Pool(8) as pool:
        results = list(tqdm(pool.imap(process_patient_stay, stay_to_ids_dict.keys()), total=len(stay_to_ids_dict)))

    min_max_dicts = [r[0] for r in results]
    tdigests_list = [r[1] for r in results]

    # Merge all min_max_dicts
    min_max_dict = merge_min_max_dicts(min_max_dicts)

    # Merge t-digests from all chunks
    final_tdigests = merge_tdigests(tdigests_list)

    percentile_dict = {key: {
        "1": final_tdigests[key].percentile(1),
        "2": final_tdigests[key].percentile(2),
        "98": final_tdigests[key].percentile(98),
        "99": final_tdigests[key].percentile(99),
    } for key in final_tdigests.keys()}

    to_delete = []
    new_elems = {}
    for key, value in percentile_dict.items():
        if '___' in key:
            key1, key2 = key.split('___')
            if key1 not in new_elems:
                new_elems[key1] = {}
            new_elems[key1][key2] = value
            to_delete.append(key)

    for key in to_delete:
        del percentile_dict[key]

    for key, value in new_elems.items():
        percentile_dict[key] = value

    with open(f"{mimiciv_path}percentile_dict.json", "w") as f:
        json.dump(percentile_dict, f)

    print("Done with percentile dict!")

    with open(f"{mimiciv_path}min_max_dict.json", "w") as f:
        json.dump(min_max_dict, f)

    print("Done with min_max dict!")
