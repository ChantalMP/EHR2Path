import datetime as dt
import gc
import gzip
import json
import logging
import multiprocessing
import os
import pickle
import random
import re
import time
from collections import defaultdict

import warnings
from functools import partial
from multiprocessing import Pool

from pandas.errors import SettingWithCopyWarning


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

import numpy as np
import pandas as pd
from dask import dataframe as dd
from tqdm import tqdm

from tqdm.contrib.concurrent import process_map
from mimic_iv_extraction.paths import mimiciv_path as mimiciv_path
from mimic_iv_extraction.paths import mimiciv_note_path, mimiciv_imgcxr_path
from mimic_iv_extraction.paths import mimic_iv_ed_path as mimiciv_ed_path
from mimic_iv_extraction.utils import convert_ts_to_columns, create_hourly_event_df, ndc_meds, noon_time, mean_or_last_agg, \
    convert_table_to_sorted_periods_string, convert_column_to_sorted_periods_string, convert_microbiology, convert_column_to_time_events_string, \
    extract_impression
from patient_model.utils import get_icd_category


def load_mimiciv(mimiciv_path, mimiciv_imgcxr_path, mimiciv_note_path, mimiciv_ed_path):
    # Inputs:
    #   mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #   mimiciv_imgcxr_path -> Path to MIMIC IV image data (CXR)
    #   mimiciv_note_path -> Path to MIMIC IV note data
    #
    # Outputs:
    #   df's -> Many dataframes with all loaded MIMIC IV tables

    ### -> Initializations & Data Loading
    ###    Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)

    ## HOSP
    df_omr = dd.read_csv(mimiciv_path + 'hosp/omr.csv.gz', compression='gzip', assume_missing=True,
                         dtype={'result_value': 'category', 'result_name': 'category', 'seq_num': 'uint8', 'subject_id': 'int32'})
    df_admissions = dd.read_csv(mimiciv_path + 'hosp/admissions.csv.gz', assume_missing=True,
                                dtype={'subject_id':'int32','hadm_id':'int32', 'admission_type':'category','admission_location': 'category', 'discharge_location':'category',
                                       'insurance':'category', 'language':'category', 'marital_status':'category', 'race': 'category', 'hospital_expire_flag':'uint8', 'deathtime':'object'})
    df_patients = dd.read_csv(mimiciv_path + 'hosp/patients.csv.gz', assume_missing=True, dtype={'subject_id': 'int32', 'gender':'category', 'anchor_age':'uint8',
                                                                                                 'anchor_year':'int16','anchor_year_group':'category'})

    df_transfers = dd.read_csv(mimiciv_path + 'hosp/transfers.csv.gz', assume_missing=True, dtype={'subject_id':'int32', 'hadm_id':'object', 'transfer_id':'int32',
                                                                                                   'eventtype':'category','careunit': 'category', 'outtime':'object'})
    df_transfers['hadm_id'] = df_transfers['hadm_id'].fillna(pd.NA).astype('Int32')


    df_d_labitems = dd.read_csv(mimiciv_path + 'hosp/d_labitems.csv.gz', compression='gzip', assume_missing=True, dtype={'itemid':'int32', 'fluid':'category','category':'category'})
    df_d_icd_procedures = dd.read_csv(mimiciv_path + 'hosp/d_icd_procedures.csv.gz', compression='gzip', assume_missing=True,
                                      dtype={'icd_code': 'object', 'icd_version': 'category'})
    df_d_icd_diagnoses = dd.read_csv(mimiciv_path + 'hosp/d_icd_diagnoses.csv.gz', compression='gzip', assume_missing=True,
                                     dtype={'icd_code': 'object', 'icd_version': 'category'})
    df_d_hcpcs = dd.read_csv(mimiciv_path + 'hosp/d_hcpcs.csv.gz', compression='gzip', assume_missing=True, dtype={'category': 'category', 'long_description': 'object'})
    df_diagnoses_icd = dd.read_csv(mimiciv_path + 'hosp/diagnoses_icd.csv.gz', compression='gzip', assume_missing=True,
                                   dtype={'subject_id':'int32','hadm_id':'int32','seq_num':'uint8','icd_code': 'category', 'icd_version': 'category'})
    df_drgcodes = dd.read_csv(mimiciv_path + 'hosp/drgcodes.csv.gz', compression='gzip', assume_missing=True, dtype={'subject_id':'int32','hadm_id':'int32','drg_type':'category',
                                                                                                                     'drg_code':'uint64','description':'object', 'description':'category',
                                                                                                                     'drg_severity':'category','drg_mortality':'category'})
    df_labevents = dd.read_csv(mimiciv_path + 'hosp/labevents.csv.gz', compression='gzip', assume_missing=True,
                               dtype={'labevent_id':'int32','subject_id':'int32','specimen_id':'int32','itemid':'int32', 'order_provider_id':'object',
                                      'storetime': 'object', 'value': 'category', 'valueuom': 'category', 'flag': 'category', 'priority': 'category','comments': 'category'})
    df_labevents['hadm_id'] = df_labevents['hadm_id'].fillna(pd.NA).astype('Int32')

    df_microbiologyevents = dd.read_csv(mimiciv_path + 'hosp/microbiologyevents.csv.gz', compression='gzip', assume_missing=True,
                                        dtype={'microevent_id': 'int32', 'subject_id': 'int32', 'micro_specimen_id': 'int32', 'spec_itemid': 'int32',
                                               'order_provider_id': 'object', 'spec_type_desc': 'category', 'test_seq': 'uint8', 'test_name': 'category',
                                               'org_name': 'category', 'quantity': 'category', 'ab_name': 'category', 'dilution_text': 'category',
                                               'dilution_comparison': 'category', 'dilution_value': 'category', 'interpretation': 'category',
                                               'comments': 'category'})

    # convert isolate_num to int16
    df_microbiologyevents['isolate_num'] = df_microbiologyevents['isolate_num'].fillna(pd.NA).astype('Int16')
    df_microbiologyevents['hadm_id'] = df_microbiologyevents['hadm_id'].fillna(pd.NA).astype('Int32')
    df_microbiologyevents['ab_itemid'] = df_microbiologyevents['ab_itemid'].fillna(pd.NA).astype('Int32')
    df_microbiologyevents['org_itemid'] = df_microbiologyevents['org_itemid'].fillna(pd.NA).astype('Int32')
    df_microbiologyevents['org_name'] = df_microbiologyevents['org_name'].cat.add_categories('no growth')
    df_microbiologyevents['ab_name'] = df_microbiologyevents['ab_name'].cat.add_categories('none')
    df_microbiologyevents['org_name'] = df_microbiologyevents['org_name'].fillna('no growth')
    df_microbiologyevents['ab_name'] = df_microbiologyevents['ab_name'].fillna('none')
    df_microbiologyevents['test_itemid'] = df_microbiologyevents['test_itemid'].fillna(pd.NA).astype('Int32')

    df_prescriptions = dd.read_csv(mimiciv_path + 'hosp/prescriptions.csv.gz', compression='gzip', assume_missing=True,
                                   dtype={'subject_id':'int32','hadm_id':'int32', 'pharmacy_id':'int32', 'order_provider_id':'category', 'drug_type':'category', 'drug':'category',
                                          'formulary_drug_cd':'category', 'gsn':'category', 'prod_strength':'category', 'form_rx':'category',
                                          'dose_val_rx':'category', 'dose_unit_rx':'category', 'form_val_disp':'category', 'form_unit_disp':'category','doses_per_24_hrs':'float32', 'route':'category'})
    df_prescriptions['doses_per_24_hrs'] = df_prescriptions['doses_per_24_hrs'].astype('float64').round(2)
    # convert poe_seq to int32
    df_prescriptions['poe_seq'] = df_prescriptions['poe_seq'].fillna(pd.NA).astype('Int32')

    df_procedures_icd = dd.read_csv(mimiciv_path + 'hosp/procedures_icd.csv.gz', compression='gzip', assume_missing=True,
                                    dtype={'subject_id':'int32','hadm_id':'int32','seq_num':'uint8','icd_code': 'category', 'icd_version': 'category'})
    df_services = dd.read_csv(mimiciv_path + 'hosp/services.csv.gz', compression='gzip', assume_missing=True, dtype={'subject_id':'int32','hadm_id':'int32','prev_service':'category', 'curr_service':'category'})

    ## ICU
    df_d_items = dd.read_csv(mimiciv_path + 'icu/d_items.csv.gz', compression='gzip', assume_missing=True, dtype={'itemid': 'int32', 'linksto':'category', 'category':'category',
                                                                                                                  'unitname':'category', 'param_type':'category', 'lownormalvalue':'category', 'highnormalvalue':'category'})
    df_procedureevents = dd.read_csv(mimiciv_path + 'icu/procedureevents.csv.gz', compression='gzip', assume_missing=True,
                                     dtype={'subject_id': 'int32', 'hadm_id': 'int32', 'stay_id': 'int32', 'caregiver_id': 'object', 'itemid': 'int32',
                                            'value':'float32', 'valueuom':'category', 'location':'category', 'locationcategory':'category', 'orderid':'int32',
                                            'linkorderid':'int32', 'ordercategoryname':'category', 'ordercategorydescription':'category', 'patientweight':'category',
                                            'isopenbag':'category', 'continueinnextdept':'category', 'statusdescription':'category', 'originalamount':'float32', 'originalrate':'float32'})
    df_procedureevents['value'] = df_procedureevents['value'].astype('float64').round(2)
    df_procedureevents['originalamount'] = df_procedureevents['originalamount'].astype('float64').round(2)
    df_procedureevents['originalrate'] = df_procedureevents['originalrate'].astype('float64').round(2)
    # convert caregiver_id to int32
    df_procedureevents['caregiver_id'] = df_procedureevents['caregiver_id'].fillna(pd.NA).astype('Int32')
    df_outputevents = dd.read_csv(mimiciv_path + 'icu/outputevents.csv.gz', compression='gzip', assume_missing=True, dtype={'subject_id': 'int32', 'hadm_id': 'int32', 'stay_id': 'int32', 'caregiver_id': 'int32',
                                                                                                                            'itemid':'int32', 'value':'float32', 'valueuom':'category'})
    df_outputevents['value'] = df_outputevents['value'].astype('float64').round(2)
    df_inputevents = dd.read_csv(mimiciv_path + 'icu/inputevents.csv.gz', compression='gzip', assume_missing=True,
                                 dtype={'subject_id': 'int32', 'hadm_id': 'int32', 'stay_id': 'int32', 'caregiver_id': 'int32','itemid':'int32',
                                        'amount':'float32', 'amountuom':'category', 'rate':'float32', 'rateuom':'category', 'ordercategoryname':'category',
                                        'secondaryordercategoryname':'category', 'ordercomponenttypedescription':'category', 'ordercategorydescription':'category',
                                        'patientweight': 'category', 'totalamount':'category', 'totalamountuom':'category', 'isopenbag':'category', 'continueinnextdept':'category',
                                        'statusdescription':'category', 'originalamount':'float32', 'originalrate':'float32'})
    # round amount to 2 digits after the comma and cast to best datatype
    df_inputevents['amount'] = df_inputevents['amount'].astype('float64').round(2)
    df_inputevents['rate'] = df_inputevents['rate'].astype('float64').round(2)
    df_inputevents['originalamount'] = df_inputevents['originalamount'].astype('float64').round(2)
    df_inputevents['originalrate'] = df_inputevents['originalrate'].astype('float64').round(2)
    df_icustays = dd.read_csv(mimiciv_path + 'icu/icustays.csv.gz', compression='gzip', assume_missing=True, dtype={'subject_id': 'int32', 'hadm_id': 'int32', 'stay_id': 'int32',
                                                                                                                    'first_careunit': 'category', 'last_careunit': 'category', 'los': 'float32'})
    # convert los to int16 dtype
    df_icustays['los'] = df_icustays['los'].round(1).astype('int16')
    df_chartevents = dd.read_csv(mimiciv_path + 'icu/chartevents.csv.gz', compression='gzip', assume_missing=True, low_memory=False,
                                 dtype={'subject_id': 'int32', 'hadm_id': 'int32', 'stay_id': 'int32', 'itemid': 'int32',
                                        'value': 'category', 'valuenum': 'category', 'valueuom': 'category'})
    # cast warning to uint8
    df_chartevents['warning'] = df_chartevents['warning'].fillna(pd.NA).astype('Int8')
    df_chartevents['caregiver_id'] = df_chartevents['caregiver_id'].fillna(pd.NA).astype('Int32')

    ## CXR
    df_mimic_cxr_split = dd.read_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-split.csv', assume_missing=True, dtype={'subject_id':'int32','study_id': 'int32', 'split':'category'})
    df_mimic_cxr_chexpert = dd.read_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-chexpert.csv', assume_missing=True,
                                        dtype={'subject_id':'int32','study_id': 'int32', 'Atelectasis':'category', 'Cardiomegaly':'category', 'Consolidation':'category',
                                                'Edema':'category', 'Enlarged Cardiomediastinum':'category', 'Fracture':'category', 'Lung Lesion':'category',
                                                'Lung Opacity':'category', 'No Finding':'category', 'Pleural Effusion':'category', 'Pleural Other':'category',
                                                'Pneumonia':'category', 'Pneumothorax':'category', 'Support Devices':'category'})
    try:
        df_mimic_cxr_metadata = dd.read_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv', assume_missing=True,
                                            dtype={'dicom_id': 'object', 'subject_id': 'int32', 'study_id': 'int32', 'PerformedProcedureStepDescription': 'category',
                                                   'ViewPosition': 'category', 'Rows': 'int16' ,'ProcedureCodeSequence_CodeMeaning':'category', 'ViewCodeSequence_CodeMeaning':'category',
                                                   'PatientOrientationCodeSequence_CodeMeaning': 'category'})
    except:
        df_mimic_cxr_metadata = pd.read_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object'})
        df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    df_mimic_cxr_negbio = dd.read_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-negbio.csv', assume_missing=True,
                                      dtype={'subject_id':'int32','study_id': 'int32', 'Atelectasis':'category', 'Cardiomegaly':'category', 'Consolidation':'category',
                                             'Edema':'category', 'Enlarged Cardiomediastinum':'category', 'Fracture':'category', 'Lung Lesion':'category',
                                             'Lung Opacity':'category', 'No Finding':'category', 'Pleural Effusion':'category', 'Pleural Other':'category',
                                             'Pneumonia':'category', 'Pneumothorax':'category', 'Support Devices':'category'})

    ## NOTES
    df_disch_notes = dd.read_csv(mimiciv_note_path + 'note/discharge.csv.gz', dtype={'subject_id':'int32', 'hadm_id': 'int32', 'note_type': 'category', 'note_seq': 'int32'})
    df_rad_notes = dd.read_csv(mimiciv_note_path + 'note/radiology.csv.gz', dtype={'subject_id':'int32','note_type':'category', 'note_seq': 'int16'})
    df_rad_notes['hadm_id'] = df_rad_notes['hadm_id'].fillna(pd.NA).astype('Int32')
    df_rad_details = dd.read_csv(mimiciv_note_path + 'note/radiology_detail.csv.gz',
                                 dtype={'subject_id': 'int32', 'field_name': 'category', 'field_value': 'category', 'field_ordinal':'int8'})

    # ED
    # df_ed_diagnosis, df_ed_stays, df_ed_medrecon, df_ed_pyxis, df_ed_triage, df_ed_vital
    df_ed_diagnosis = dd.read_csv(mimiciv_ed_path + 'ed/diagnosis.csv.gz', compression='gzip', assume_missing=True,
                                  dtype={'subject_id': 'int32', 'stay_id': 'int32', 'seq_num': 'uint8', 'icd_code': 'category', 'icd_version': 'category', 'icd_title': 'category'})
    df_ed_stays = dd.read_csv(mimiciv_ed_path + 'ed/edstays.csv.gz', compression='gzip', assume_missing=True,
                              dtype={'subject_id': 'int32', 'stay_id':'int32', 'gender': 'category', 'race': 'category', 'arrival_transport': 'category', 'disposition':'category'})  # intime, outtime
    df_ed_stays['hadm_id'] = df_ed_stays['hadm_id'].fillna(pd.NA).astype('Int32')
    df_ed_medrecon = dd.read_csv(mimiciv_ed_path + 'ed/medrecon.csv.gz', compression='gzip', assume_missing=True,
                                 dtype={'subject_id': 'int32', 'stay_id': 'int32', 'name': 'category', 'gsn': 'int64', 'ndc': 'int64', 'etc_rn': 'uint8', 'etcdescription':'category'})
    df_ed_medrecon['etccode'] = df_ed_medrecon['etccode'].fillna(pd.NA).astype('Int32')
    df_ed_pyxis = dd.read_csv(mimiciv_ed_path + 'ed/pyxis.csv.gz', compression='gzip', assume_missing=True,
                              dtype={'subject_id': 'int32', 'stay_id': 'int32', 'med_rn': 'uint8', 'name': 'category', 'gsn_rn': 'uint8'})
    df_ed_triage = dd.read_csv(mimiciv_ed_path + 'ed/triage.csv.gz', compression='gzip', assume_missing=True,
                               dtype={'subject_id': 'int32', 'stay_id': 'int32', 'temperature': 'float32', 'heartrate': 'float32', 'resprate': 'float32',
                                      'o2sat': 'float32', 'sbp': 'float64', 'dbp': 'float64', 'pain': 'category', 'acuity': 'category'})
    df_ed_triage['temperature'] = df_ed_triage['temperature'].astype('float64').round(2)
    df_ed_triage['heartrate'] = df_ed_triage['heartrate'].astype('float64').round(2)
    df_ed_triage['resprate'] = df_ed_triage['resprate'].astype('float64').round(2)
    df_ed_triage['o2sat'] = df_ed_triage['o2sat'].astype('float64').round(2)
    df_ed_triage['sbp'] = df_ed_triage['sbp'].fillna(pd.NA).map_partitions(np.floor).astype('Int32')
    df_ed_triage['dbp'] = df_ed_triage['dbp'].fillna(pd.NA).map_partitions(np.floor).astype('Int32')
    df_ed_vital = dd.read_csv(mimiciv_ed_path + 'ed/vitalsign.csv.gz', compression='gzip', assume_missing=True,
                              dtype={'subject_id': 'int32', 'stay_id': 'int32', 'temperature': 'float32', 'heartrate': 'float32', 'resprate': 'float32',
                                     'o2sat': 'float32', 'pain': 'category', 'rhythm': 'object'})
    df_ed_vital['temperature'] = df_ed_vital['temperature'].astype('float64').round(2)
    df_ed_vital['heartrate'] = df_ed_vital['heartrate'].astype('float64').round(2)
    df_ed_vital['resprate'] = df_ed_vital['resprate'].astype('float64').round(2)
    df_ed_vital['o2sat'] = df_ed_vital['o2sat'].astype('float64').round(2)
    df_ed_vital['sbp'] = df_ed_vital['sbp'].fillna(pd.NA).map_partitions(np.floor).astype('Int32')
    df_ed_vital['dbp'] = df_ed_vital['dbp'].fillna(pd.NA).map_partitions(np.floor).astype('Int32')
    ### -> Data Preparation (Create full database in dask format)
    ### Fix data type issues to allow for merging
    ## CORE

    df_omr['chartdate'] = dd.to_datetime(df_omr['chartdate'])
    df_admissions['admittime'] = dd.to_datetime(df_admissions['admittime'])
    df_admissions['admittime'] = df_admissions['admittime'].dt.round('h')
    df_admissions['dischtime'] = dd.to_datetime(df_admissions['dischtime'])
    df_admissions['dischtime'] = df_admissions['dischtime'].dt.round('h')
    df_admissions['deathtime'] = dd.to_datetime(df_admissions['deathtime'])
    df_admissions['deathtime'] = df_admissions['deathtime'].dt.round('h')
    df_admissions['edregtime'] = dd.to_datetime(df_admissions['edregtime'])
    df_admissions['edregtime'] = df_admissions['edregtime'].dt.round('h')
    df_admissions['edouttime'] = dd.to_datetime(df_admissions['edouttime'])
    df_admissions['edouttime'] = df_admissions['edouttime'].dt.round('h')

    df_transfers['intime'] = dd.to_datetime(df_transfers['intime'])
    df_transfers['intime'] = df_transfers['intime'].dt.round('h')
    df_transfers['outtime'] = dd.to_datetime(df_transfers['outtime'])
    df_transfers['outtime'] = df_transfers['outtime'].dt.round('h')

    df_patients['dod'] = dd.to_datetime(df_patients['dod'])

    ## HOSP
    df_diagnoses_icd.icd_code = df_diagnoses_icd.icd_code.str.strip()
    df_diagnoses_icd.icd_version = df_diagnoses_icd.icd_version.str.strip()
    df_d_icd_diagnoses.icd_code = df_d_icd_diagnoses.icd_code.str.strip()
    df_d_icd_diagnoses.icd_version = df_d_icd_diagnoses.icd_version.str.strip()

    df_procedures_icd.icd_code = df_procedures_icd.icd_code.str.strip()
    df_procedures_icd.icd_version = df_procedures_icd.icd_version.str.strip()
    df_d_icd_procedures.icd_code = df_d_icd_procedures.icd_code.str.strip()
    df_d_icd_procedures.icd_version = df_d_icd_procedures.icd_version.str.strip()

    df_prescriptions['starttime'] = dd.to_datetime(df_prescriptions['starttime'])
    df_prescriptions['starttime'] = df_prescriptions['starttime'].dt.round('h')
    df_prescriptions['stoptime'] = dd.to_datetime(df_prescriptions['stoptime'])
    df_prescriptions['stoptime'] = df_prescriptions['stoptime'].dt.round('h')

    df_labevents['charttime'] = dd.to_datetime(df_labevents['charttime'])
    df_labevents['charttime'] = df_labevents['charttime'].dt.round('h')
    df_labevents['storetime'] = dd.to_datetime(df_labevents['storetime'])
    df_labevents['storetime'] = df_labevents['storetime'].dt.round('h')

    df_microbiologyevents['chartdate'] = dd.to_datetime(df_microbiologyevents['chartdate'])
    df_microbiologyevents['charttime'] = dd.to_datetime(df_microbiologyevents['charttime'])
    df_microbiologyevents['charttime'] = df_microbiologyevents['charttime'].dt.round('h')
    df_microbiologyevents['storedate'] = dd.to_datetime(df_microbiologyevents['storedate'])
    df_microbiologyevents['storetime'] = dd.to_datetime(df_microbiologyevents['storetime'])
    df_microbiologyevents['storetime'] = df_microbiologyevents['storetime'].dt.round('h')

    df_services['transfertime'] = dd.to_datetime(df_services['transfertime'])
    df_services['transfertime'] = df_services['transfertime'].dt.round('h')

    ## ICU
    df_procedureevents['starttime'] = dd.to_datetime(df_procedureevents['starttime'])
    df_procedureevents['starttime'] = df_procedureevents['starttime'].dt.round('h')
    df_procedureevents['endtime'] = dd.to_datetime(df_procedureevents['endtime'])
    df_procedureevents['endtime'] = df_procedureevents['endtime'].dt.round('h')
    df_procedureevents['storetime'] = dd.to_datetime(df_procedureevents['storetime'], format='%Y-%m-%d %H:%M:%S',
                                                     exact=False)  # exact=False allows to drop milliseconds
    df_procedureevents['storetime'] = df_procedureevents['storetime'].dt.round('h')

    df_outputevents['charttime'] = dd.to_datetime(df_outputevents['charttime'])
    df_outputevents['charttime'] = df_outputevents['charttime'].dt.round('h')
    df_outputevents['storetime'] = dd.to_datetime(df_outputevents['storetime'])
    df_outputevents['storetime'] = df_outputevents['storetime'].dt.round('h')

    df_inputevents['starttime'] = dd.to_datetime(df_inputevents['starttime'])
    df_inputevents['starttime'] = df_inputevents['starttime'].dt.round('h')
    df_inputevents['endtime'] = dd.to_datetime(df_inputevents['endtime'])
    df_inputevents['endtime'] = df_inputevents['endtime'].dt.round('h')
    df_inputevents['storetime'] = dd.to_datetime(df_inputevents['storetime'])
    df_inputevents['storetime'] = df_inputevents['storetime'].dt.round('h')

    df_icustays['intime'] = dd.to_datetime(df_icustays['intime'])
    df_icustays['intime'] = df_icustays['intime'].dt.round('h')
    df_icustays['outtime'] = dd.to_datetime(df_icustays['outtime'])
    df_icustays['outtime'] = df_icustays['outtime'].dt.round('h')

    df_chartevents['charttime'] = dd.to_datetime(df_chartevents['charttime'])
    df_chartevents['charttime'] = df_chartevents['charttime'].dt.round('h')
    df_chartevents['storetime'] = dd.to_datetime(df_chartevents['storetime'])
    df_chartevents['storetime'] = df_chartevents['storetime'].dt.round('h')

    ## CXR
    if (not 'cxrtime' in df_mimic_cxr_metadata.columns) or (not 'Img_Filename' in df_mimic_cxr_metadata.columns):
        # Create CXRTime variable if it does not exist already
        print("Processing CXRtime stamps")
        df_cxr = df_mimic_cxr_metadata.compute()
        df_cxr['StudyDateForm'] = pd.to_datetime(df_cxr['StudyDate'], format='%Y%m%d')
        df_cxr['StudyTimeForm'] = df_cxr.apply(lambda x: '%#010.3f' % x['StudyTime'], 1)
        df_cxr['StudyTimeForm'] = pd.to_datetime(df_cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
        df_cxr['cxrtime'] = df_cxr.apply(lambda r: dt.datetime.combine(r['StudyDateForm'], r['StudyTimeForm']), 1)
        # Add paths and info to images in cxr
        df_mimic_cxr_jpg = pd.read_csv(
            mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-jpeg-txt.csv')
        df_cxr = pd.merge(df_mimic_cxr_jpg, df_cxr, on='dicom_id')
        # Save
        df_cxr.to_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata_updated.csv', index=False)
        # Read back the dataframe
        try:
            df_mimic_cxr_metadata = dd.read_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata_updated.csv', assume_missing=True,
                                                dtype={'dicom_id': 'object', 'Note': 'object'}, blocksize=None)
        except:
            df_mimic_cxr_metadata = pd.read_csv(mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata_updated.csv',
                                                dtype={'dicom_id': 'object', 'Note': 'object'})
            df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    df_mimic_cxr_metadata['cxrtime'] = dd.to_datetime(df_mimic_cxr_metadata['cxrtime'])
    df_mimic_cxr_metadata['cxrtime'] = df_mimic_cxr_metadata['cxrtime'].dt.round('h')

    ## NOTES
    df_disch_notes['charttime'] = dd.to_datetime(df_disch_notes['charttime'])
    df_disch_notes['charttime'] = df_disch_notes['charttime'].dt.round('h')
    df_disch_notes['storetime'] = dd.to_datetime(df_disch_notes['storetime'])
    df_disch_notes['storetime'] = df_disch_notes['storetime'].dt.round('h')

    df_rad_notes['charttime'] = dd.to_datetime(df_rad_notes['charttime'])
    df_rad_notes['charttime'] = df_rad_notes['charttime'].dt.round('h')
    df_rad_notes['storetime'] = dd.to_datetime(df_rad_notes['storetime'])
    df_rad_notes['storetime'] = df_rad_notes['storetime'].dt.round('h')

    ## ED
    df_ed_stays['intime'] = dd.to_datetime(df_ed_stays['intime'])
    df_ed_stays['intime'] = df_ed_stays['intime'].dt.round('h')
    df_ed_stays['outtime'] = dd.to_datetime(df_ed_stays['outtime'])
    df_ed_stays['outtime'] = df_ed_stays['outtime'].dt.round('h')
    df_ed_medrecon['charttime'] = dd.to_datetime(df_ed_medrecon['charttime'])
    df_ed_medrecon['charttime'] = df_ed_medrecon['charttime'].dt.round('h')
    df_ed_pyxis['charttime'] = dd.to_datetime(df_ed_pyxis['charttime'])
    df_ed_pyxis['charttime'] = df_ed_pyxis['charttime'].dt.round('h')
    df_ed_vital['charttime'] = dd.to_datetime(df_ed_vital['charttime'])
    df_ed_vital['charttime'] = df_ed_vital['charttime'].dt.round('h')
    ### -> SORT data
    ## CORE
    print('PROCESSING "CORE" DB...')
    print("omr")
    df_omr = df_omr.compute().sort_values(by=['subject_id'])
    print("admissions")
    df_admissions = df_admissions.compute().sort_values(by=['subject_id', 'hadm_id'])
    print("patients")
    df_patients = df_patients.compute().sort_values(by=['subject_id'])
    print("transfers")
    df_transfers = df_transfers.compute().sort_values(by=['subject_id', 'hadm_id'])

    ## HOSP
    print('PROCESSING "HOSP" DB...')
    print("diagnoses_icd")
    df_diagnoses_icd = df_diagnoses_icd.compute().sort_values(by=['subject_id'])
    print("drgcodes")
    df_drgcodes = df_drgcodes.compute().sort_values(by=['subject_id', 'hadm_id'])
    print("labevents")
    df_labevents = df_labevents.compute().sort_values(by=['subject_id', 'hadm_id'])
    print("microbiologyevents")
    df_microbiologyevents = df_microbiologyevents.compute().sort_values(by=['subject_id', 'hadm_id'])
    print("prescriptions")
    df_prescriptions = df_prescriptions.compute().sort_values(by=['subject_id', 'hadm_id'])
    df_prescriptions = ndc_meds(df_prescriptions)  # add non-proprietary drug names
    print("procedures_icd")
    df_procedures_icd = df_procedures_icd.compute().sort_values(by=['subject_id', 'hadm_id'])
    print("services")
    df_services = df_services.compute().sort_values(by=['subject_id', 'hadm_id'])
    # --> Unwrap dictionaries
    print("d_icd_diagnoses")
    df_d_icd_diagnoses = df_d_icd_diagnoses.compute()
    print("d_icd_procedures")
    df_d_icd_procedures = df_d_icd_procedures.compute()
    print("d_hcpcs")
    df_d_hcpcs = df_d_hcpcs.compute()
    print("d_labitems")
    df_d_labitems = df_d_labitems.compute()

    ## ICU
    print('PROCESSING "ICU" DB...')
    print("procedureevents")
    df_procedureevents = df_procedureevents.compute().sort_values(by=['subject_id', 'hadm_id', 'stay_id'])

    print("outputevents")
    df_outputevents = df_outputevents.compute().sort_values(by=['subject_id', 'hadm_id', 'stay_id'])
    print("inputevents")
    df_inputevents = df_inputevents.compute().sort_values(by=['subject_id', 'hadm_id', 'stay_id'])
    print("icustays")
    df_icustays = df_icustays.compute().sort_values(by=['subject_id', 'hadm_id', 'stay_id'])
    print("chartevents")
    df_chartevents = df_chartevents.compute().sort_values(by=['subject_id', 'hadm_id', 'stay_id'])
    # --> Unwrap dictionaries
    print("d_items")
    df_d_items = df_d_items.compute()

    ## CXR
    print('PROCESSING "CXR" DB...')
    print("mimic_cxr_split")
    df_mimic_cxr_split = df_mimic_cxr_split.compute().sort_values(by=['subject_id'])
    print("mimic_cxr_chexpert")
    df_mimic_cxr_chexpert = df_mimic_cxr_chexpert.compute().sort_values(by=['subject_id'])
    print("mimic_cxr_metadata")
    df_mimic_cxr_metadata = df_mimic_cxr_metadata.compute().sort_values(by=['subject_id'])
    print("mimic_cxr_negbio")
    df_mimic_cxr_negbio = df_mimic_cxr_negbio.compute().sort_values(by=['subject_id'])

    ## NOTES
    print('PROCESSING "NOTES" DB...')
    print("disch_notes")
    df_disch_notes = df_disch_notes.compute().sort_values(by=['subject_id', 'hadm_id'])
    print("rad_notes")
    df_rad_notes = df_rad_notes.compute().sort_values(by=['subject_id', 'hadm_id'])
    print("rad_details")
    df_rad_details = df_rad_details.compute().sort_values(by=['subject_id'])

    ## ED
    print('PROCESSING "ED" DB...')
    print("ed_diagnosis")
    df_ed_diagnosis = df_ed_diagnosis.compute().sort_values(by=['subject_id', 'stay_id'])
    print("ed_stays")
    df_ed_stays = df_ed_stays.compute().sort_values(
        by=['subject_id', 'hadm_id', 'stay_id'])  # can be linked to MIMIC-IV hospital admission, if NULL patient was not admitted
    print("ed_medrecon")
    df_ed_medrecon = df_ed_medrecon.compute().sort_values(by=['subject_id', 'stay_id'])
    print("ed_pyxis")
    df_ed_pyxis = df_ed_pyxis.compute().sort_values(by=['subject_id', 'stay_id'])
    print("ed_triage")
    df_ed_triage = df_ed_triage.compute().sort_values(by=['subject_id', 'stay_id'])
    print("ed_vital")
    df_ed_vital = df_ed_vital.compute().sort_values(by=['subject_id', 'stay_id'])

    # Return
    return (df_omr, df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_drgcodes, df_labevents, df_microbiologyevents, df_prescriptions, df_procedures_icd, df_services, df_d_icd_diagnoses, df_d_icd_procedures, df_d_hcpcs, df_d_labitems, df_procedureevents, df_outputevents, df_inputevents,
            df_icustays, df_chartevents, df_d_items, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_metadata, df_mimic_cxr_negbio, df_disch_notes, df_rad_notes, df_rad_details, df_ed_diagnosis, df_ed_stays, df_ed_medrecon, df_ed_pyxis, df_ed_triage, df_ed_vital)


class Patient_ADM(object):
    def __init__(self, omr=None, admissions=None, patients=None, transfers=None, core=None,
                 diagnoses_icd=None, drgcodes=None, labevents=None, microbiologyevents=None,
                 prescriptions=None, procedures_icd=None, services=None, disch_notes=None, rad_notes=None):
        ## CORE
        self.hour_time = None
        self.omr = omr
        self.admissions = admissions
        self.patients = patients
        self.transfers = transfers
        self.core = core
        ## HOSP
        self.diagnoses_icd = diagnoses_icd
        self.drgcodes = drgcodes
        self.labevents = labevents
        self.microbiologyevents = microbiologyevents
        self.prescriptions = prescriptions
        self.procedures_icd = procedures_icd
        self.services = services
        ## NOTES
        self.disch_notes = disch_notes
        self.rad_notes = rad_notes

        if self.admissions is not None:
            self.admittime = self.admissions['admittime'].values[0]
            self.dischtime = self.admissions['dischtime'].values[0]
            self.stayhours = (self.dischtime - self.admittime) / pd.Timedelta(hours=1)

        else:
            self.admittime = None
            self.dischtime = None
            self.stayhours = None

    def pre_process(self):

        if self.prescriptions is not None:
            self.prescriptions = self.prescriptions.sort_values(by=['time'])

        if self.procedures_icd is not None:
            self.procedures_icd['chartdate'] = pd.to_datetime(self.procedures_icd['chartdate'])
            self.procedures_icd = self.procedures_icd.sort_values(by=['chartdate'])

        if self.rad_notes is not None:
            self.rad_notes = self.rad_notes.sort_values(by=['charttime'])
            # drop all samples where 'exam_name' == 'NaN' (usually addendums, were already merged)
            self.rad_notes = self.rad_notes.dropna(subset=['exam_name'])

        self.icd_mapping = pd.read_csv('mimic_iv_extraction/mappings/ICD9_to_ICD10_mapping.txt', header=0, delimiter="\t")
        self.icd_categories = self.get_icd_categories()

        self.admittime = self.admissions['admittime'].values[0]
        self.dischtime = self.admissions['dischtime'].values[0]
        self.stayhours = (self.dischtime - self.admittime) / pd.Timedelta(hours=1)

    def extract_hosp_diag(self):
        if self.disch_notes is not None and len(self.disch_notes.index) > 0:
            disch_summary = self.disch_notes['text'].values[0]
            # extract diagnosis from discharge summary
            pattern = r"Discharge Diagnosis:\s+(.*?)(?=(Discharge Condition|Discharge Diagnosis))"
            matches = re.findall(pattern, disch_summary, re.DOTALL)
            if matches:
                extracted_diag = matches[-1][0].strip().replace("\n", ",")
            else:
                extracted_diag = "No discharge diagnosis available"
        else:
            extracted_diag = "No diagnosis available"
        return extracted_diag

    def get_icd_categories(self):
        if self.diagnoses_icd is not None and len(self.diagnoses_icd.index) > 0:
            # get all codes in icd_code where icd_version == 9
            diag_codes_idc9 = self.diagnoses_icd[self.diagnoses_icd.icd_version == '9'].icd_code.values
            diag_codes_idc10 = self.diagnoses_icd[self.diagnoses_icd.icd_version == '10'].icd_code.values
            # map ICD10 to ICD9
            diag_codes_idc10_mapped = self.icd_mapping[self.icd_mapping.icd10cm.isin(diag_codes_idc10)].icd9cm.values

            diag_codes = np.concatenate([diag_codes_idc9, diag_codes_idc10_mapped])
            icd_categories = get_icd_category(diag_codes)
            icd_categories = ';'.join(icd_categories)

        else:
            icd_categories = 'unknown'

        return icd_categories

    def get_data_until_hour(self, hour_idx, hour_time=None, restrict_to_N_hours=None):
        # Get data for the hour_idx
        if hour_time is None:
            hour_time = self.admittime + pd.Timedelta(hours=hour_idx)
        self.transfers = self.transfers[(self.transfers['time'] <= hour_time)]
        self.services = self.services[(self.services['time'] <= hour_time)] if self.services is not None else None
        self.omr = self.omr[self.omr.chartdate <= hour_time] if self.omr is not None else None  # chartdate is always 00:00, so it will be included once the day is started
        self.labevents = self.labevents[self.labevents.charttime <= hour_time] if self.labevents is not None else None
        self.microbiologyevents = self.microbiologyevents[self.microbiologyevents.charttime <= hour_time] if self.microbiologyevents is not None else None
        self.prescriptions = self.prescriptions[self.prescriptions.time <= hour_time] if self.prescriptions is not None else None
        if hour_idx == -1:  # prior to admission, only include procedures_icd that happened on prior days (should not happen)
            self.procedures_icd = self.procedures_icd[self.procedures_icd.chartdate.dt.date < hour_time.date()] if self.procedures_icd is not None else None
        else:
            self.procedures_icd = self.procedures_icd[self.procedures_icd.chartdate <= hour_time] if self.procedures_icd is not None else None
        self.rad_notes = self.rad_notes[self.rad_notes.charttime <= hour_time] if self.rad_notes is not None else None

        if restrict_to_N_hours is not None:
            start_hour_time = hour_time - pd.Timedelta(hours=restrict_to_N_hours)
            self.transfers = self.transfers[(self.transfers['time'] >= start_hour_time)]
            self.services = self.services[(self.services['time'] > start_hour_time)] if self.services is not None else None
            self.omr = self.omr[self.omr.chartdate.dt.date >= start_hour_time.date()] if self.omr is not None else None  # chartdate is always 00:00, so it will be included once the day is started
            self.labevents = self.labevents[self.labevents.charttime > start_hour_time] if self.labevents is not None else None
            self.microbiologyevents = self.microbiologyevents[self.microbiologyevents.charttime > start_hour_time] if self.microbiologyevents is not None else None
            self.prescriptions = self.prescriptions[self.prescriptions.time > start_hour_time] if self.prescriptions is not None else None
            self.procedures_icd = self.procedures_icd[self.procedures_icd.chartdate.dt.date >= start_hour_time.date()] if self.procedures_icd is not None else None #include if on day with included hours
            self.rad_notes = self.rad_notes[self.rad_notes.charttime > start_hour_time] if self.rad_notes is not None else None

        if self.dischtime is None or hour_time < self.dischtime:  # diagnosis not yet available
            self.diag_avail = False
            self.disch_notes = None
            self.admissions.dischtime = None
            self.admissions.deathtime = None
            self.diagnosis_icd = None
            self.disch_notes = None
            self.drgcodes = None
            self.icd_categories = None
            self.stayhours = None

        return hour_time

    def get_data_at_hour(self, hour_idx):
        # Get data for the hour_idx
        hour_time = self.admittime + pd.Timedelta(hours=hour_idx)
        self.transfers = self.transfers[(self.transfers['time'] == hour_time)]
        self.services = self.services[(self.services['time'] == hour_time)] if self.services is not None else None
        # include daily things always at 00:00 midnight of the day they will happen or at admittime for admission day
        if hour_idx == 0:  # admission day - include all from that day at admission
            self.omr = self.omr[(self.omr.chartdate.dt.date == hour_time.date()) & (
                    self.omr.chartdate <= hour_time)] if self.omr is not None else None  # chartdate is always 00:00, so it will be included once the day is started
        else:
            self.omr = self.omr[self.omr.chartdate == hour_time] if self.omr is not None else None
        self.labevents = self.labevents[self.labevents.charttime == hour_time] if self.labevents is not None else None
        self.microbiologyevents = self.microbiologyevents[self.microbiologyevents.charttime == hour_time] if self.microbiologyevents is not None else None
        self.prescriptions = self.prescriptions[self.prescriptions.time == hour_time] if self.prescriptions is not None else None
        if hour_idx == 0:
            self.procedures_icd = self.procedures_icd[
                (self.procedures_icd.chartdate.dt.date == hour_time.date()) & (self.procedures_icd.chartdate <= hour_time)] if self.procedures_icd is not None else None
        else:
            self.procedures_icd = self.procedures_icd[self.procedures_icd.chartdate == hour_time] if self.procedures_icd is not None else None
        self.rad_notes = self.rad_notes[self.rad_notes.charttime == hour_time] if self.rad_notes is not None else None
        return hour_time

    def get_description(self, hour_time, summary_level=0, out_time=None, hospital_los=None, los_only=False, force_los=False):
        # if force_los is True, don't add LOS randomly but always -> used for Hospital discharge evaluation start - iterative evals can just add hospital_los, and training sets use randomness
        assert hour_time is not None, "hour_time must be set"
        description_yaml = {}
        if hospital_los is not None:
            description_yaml['LOS'] = f"{hospital_los}"
        else: #in validation out_time is always None, so GT LOS is never used (except for ICD prediction at discharge)
            hours_to_discharge = (out_time - hour_time) / pd.Timedelta(hours=1) if out_time is not None else None
            if hours_to_discharge is not None and hours_to_discharge >= 0:
                if not force_los and random.random() < 0.5:  # delete in 50% of cases to teach model to predict los even if it is not available
                    pass
                else:
                    # add noise and add to description
                    # noise pattern: randomly select hour between -10% to +10% of LOS
                    noised_hours_to_discharge = round(random.uniform(0.8, 1.2) * hours_to_discharge)
                    description_yaml['LOS'] = f"{int(noised_hours_to_discharge)} hours"

        if los_only:
            return description_yaml

        description_yaml['General'] = (
            f"{self.admissions['admission_type'].values[0].lower()} patient, {int(self.patients['anchor_age'])}-year old {'female' if self.patients['gender'].values[0] == 'F' else 'male'}, "
            f"insurance: {self.admissions['insurance'].values[0]}, {self.admissions['race'].values[0].lower()}, language: {self.admissions['language'].values[0].lower()}")
        if self.transfers is not None and len(self.transfers.index) > 0:
            description_yaml['Patient Location'] = convert_table_to_sorted_periods_string(df=self.transfers, curr_time=hour_time)
        if self.services is not None:
            caretaker = convert_table_to_sorted_periods_string(df=self.services, curr_time=hour_time, use_service_mapping=True)
            if caretaker != "":
                description_yaml['Care Taker'] = caretaker

        # omr
        if self.omr is not None and len(self.omr.index) > 0:
            description_yaml['Outpatient Measurements'] = {}
            for col in self.omr.columns[2:]:
                value = self.omr[col]
                if not pd.isna(value).all():
                    # iterate over not nan values and how many hours ago they were measured
                    restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                    ts_values = convert_column_to_time_events_string(self.omr, value, hour_time, category=None, time_col_name='chartdate', use_days=True, restrict_to_N=restrict_to_N)
                    description_yaml['Outpatient Measurements'][f"{col}"] = ts_values

        # labevents
        if self.labevents is not None and len(self.labevents.index) > 0:
            description_yaml['Lab Results'] = {}
            lab_cols = [col for col in self.labevents.columns[1:] if
                        not col.endswith('_valueuom') and not col.endswith('_ref_range_lower') and not col.endswith('_ref_range_upper') and not col.endswith('_flag') and not col.endswith(
                            '_priority') and not col.endswith('_comments')]
            for col in lab_cols:
                values = self.labevents[col]
                if not pd.isna(values).all():
                    # iterate over not nan values and how many hours ago they were measured
                    restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                    ts_values = convert_column_to_time_events_string(self.labevents, values, hour_time, col=col, restrict_to_N=restrict_to_N)

                    ref_range_lower = self.labevents[f"{col}_ref_range_lower"].dropna().values if f"{col}_ref_range_lower" in self.labevents.columns else []
                    ref_range_upper = self.labevents[f"{col}_ref_range_upper"].dropna().values if f"{col}_ref_range_upper" in self.labevents.columns else []
                    ref_range_lower = ref_range_lower[0] if len(ref_range_lower) > 0 else None
                    ref_range_upper = ref_range_upper[0] if len(ref_range_upper) > 0 else None
                    unit = self.labevents[f"{col}_valueuom"].dropna().values if f"{col}_valueuom" in self.labevents.columns else []
                    unit = unit[0] if len(unit) > 0 else None
                    unit_str = ""
                    if not pd.isna(unit):
                        unit_str += f"{unit}"
                    if not pd.isna(ref_range_lower) and not pd.isna(ref_range_upper):
                        if unit_str != "":
                            unit_str += ", "
                        unit_str += f"normal range: {ref_range_lower}-{ref_range_upper}"
                    if col == 'Estimated GFR (MDRD equation)':
                        unit_str = "mL/min/1.73 m2, <60 = Chronic Kidney Disease, <15 = Kidney Failure"

                    if unit_str == "":
                        descriptor = f"{col}"
                    else:
                        descriptor = f"{col} ({unit_str})"

                    description_yaml['Lab Results'][f"{descriptor}"] = ts_values

        # microbiologyevents
        if self.microbiologyevents is not None and len(self.microbiologyevents.index) > 0:
            description_yaml['Microbiology Growth Results'] = {}
            performed_tests = defaultdict(list)
            for row in self.microbiologyevents.iterrows():
                hours_ago = int((hour_time - row[1].charttime) / pd.Timedelta(hours=1))
                growth_desc = f"{row[1]['org_name']}"
                if 'ab_interpretation' in row[1]:
                    ab_desc = f" antibiotics: {row[1]['ab_interpretation']}" if not pd.isna(row[1]['ab_interpretation']) else ""
                    performed_tests[f"{row[1].test_name} - {row[1].spec_type_desc.lower()}"].append(f"{hours_ago}: {growth_desc}{ab_desc}")
                else:
                    performed_tests[f"{row[1].test_name} - {row[1].spec_type_desc.lower()}"].append(f"{growth_desc}")
            for test_name, values in performed_tests.items():
                description_yaml['Microbiology Growth Results'][test_name] = ', '.join(values)

        # prescriptions
        if self.prescriptions is not None and len(self.prescriptions.index) > 0:
            description_yaml['Prescriptions'] = {}
            pred_columns = [col for col in self.prescriptions.columns[1:] if col is not None and not col.endswith('_dose_val_rx') and not col.endswith('_dose_unit_rx')]
            for col, values in self.prescriptions[pred_columns].items():
                values = self.prescriptions[['time', col]]
                restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                ts_values = convert_column_to_sorted_periods_string(values, hour_time, time_col_name='time', col=col, check_zero=True, restrict_to_N=restrict_to_N)
                if ts_values != "":
                    description_yaml['Prescriptions'][col] = ts_values

        # procedures
        if self.procedures_icd is not None and len(self.procedures_icd.index) > 0:
            description_yaml['Procedures'] = {}
            for row in self.procedures_icd.iterrows():
                days_ago = int((hour_time - row[1].chartdate) / pd.Timedelta(days=1))
                if f"{days_ago} days" in description_yaml['Procedures']:
                    description_yaml['Procedures'][f"{days_ago} days"] += "; " + row[1].long_title  # + f" (ICD-{row[1].icd_version}: {row[1].icd_code})"
                else:
                    description_yaml['Procedures'][f"{days_ago} days"] = row[1].long_title

        # rad_notes
        if self.rad_notes is not None and len(self.rad_notes.index) > 0:
            description_yaml['Radiology Notes'] = {}
            if summary_level >= 1:  # keep only impression section
                if summary_level < 3:
                    for row in self.rad_notes.iterrows():
                        hours_ago = int((hour_time - row[1].charttime) / pd.Timedelta(hours=1))
                        impression = extract_impression(row[1].text)
                        if f"{hours_ago}" in description_yaml['Radiology Notes']: #already another report at same hour -> list them
                            description_yaml['Radiology Notes'][f"{hours_ago}"] += "; " + row[1]['exam_name'] + ": " + impression.replace('\n', ' ')
                        else:
                            description_yaml['Radiology Notes'][f"{hours_ago}"] = row[1]['exam_name'] + ": " + impression.replace('\n', ' ')
                else:
                    # keep 1 report per modality
                    modalities = []
                    # iterate from back to front to keep only the last report per modality
                    for row in self.rad_notes[::-1].iterrows():
                        modality = row[1]['exam_name']
                        if modality not in modalities:
                            modalities.append(modality)
                            hours_ago = int((hour_time - row[1].charttime) / pd.Timedelta(hours=1))
                            impression = extract_impression(row[1].text)
                            if f"{hours_ago}" in description_yaml['Radiology Notes']:
                                description_yaml['Radiology Notes'][f"{hours_ago}"] += "; " + row[1]['exam_name'] + ": " + impression.replace('\n', ' ')
                            else:
                                description_yaml['Radiology Notes'][f"{hours_ago}"] = row[1]['exam_name'] + ": " + impression.replace('\n', ' ')

            else:
                for row in self.rad_notes.iterrows():
                    hours_ago = int((hour_time - row[1].charttime) / pd.Timedelta(hours=1))
                    description_yaml['Radiology Notes'][f"{hours_ago}"] = row[1]['exam_name'] + ": " + row[1].text.replace('\n', ' ')

        return description_yaml

    def get_change_description(self, entered_icu, out_time=None, next_hour_time=None, hospital_los=None):
        description_yaml = {}
        if hospital_los is not None:
            description_yaml['LOS'] = f"{hospital_los}"
        else:
            hours_to_discharge = (out_time - next_hour_time) / pd.Timedelta(hours=1) if out_time is not None else None
            if hours_to_discharge is not None and hours_to_discharge >= 0:
                description_yaml['LOS'] = f"{int(hours_to_discharge)} hours"

        if self.transfers is not None and len(self.transfers.index) > 0:
            # get col name where value is 1
            all_vals = self.transfers.columns[1:][self.transfers.iloc[-1, 1:].astype(bool)].values
            if len(all_vals) > 0:
                description_yaml['Patient Location'] = all_vals[0]
        if self.services is not None and len(self.services.index) > 0:
            all_vals = self.services.columns[1:][self.services.iloc[-1, 1:].astype(bool)].values
            if len(all_vals) > 0:
                description_yaml['Care Taker'] = all_vals[0]

        if entered_icu:
            description_yaml['Disposition'] = "ADMITTED TO ICU"

        # labevents
        if self.labevents is not None and len(self.labevents.index) > 0:
            description_yaml['Lab Results'] = {}
            lab_cols = [col for col in self.labevents.columns[1:] if
                        not col.endswith('_valueuom') and not col.endswith('_ref_range_lower') and not col.endswith('_ref_range_upper') and not col.endswith('_flag') and not col.endswith(
                            '_priority') and not col.endswith('_comments')]
            for col in lab_cols:
                values = self.labevents[col]
                if not pd.isna(values).all():
                    description_yaml['Lab Results'][f"{col}"] = str(values.values[0])

        # microbiologyevents
        if self.microbiologyevents is not None and len(self.microbiologyevents.index) > 0:
            description_yaml['Microbiology Growth Results'] = {}
            performed_tests = defaultdict(list)
            for row in self.microbiologyevents.iterrows():
                # ab_desc = f" antibiotics: {row[1]['ab_interpretation']}" if not pd.isna(row[1]['ab_interpretation']) else ""
                growth_desc = f"{row[1]['org_name']}"
                performed_tests[f"{row[1].test_name} - {row[1].spec_type_desc.lower()}"].append(f"{growth_desc}")
            for test_name, values in performed_tests.items():
                description_yaml['Microbiology Growth Results'][test_name] = ', '.join(values)

        # prescriptions
        if self.prescriptions is not None and len(self.prescriptions.index) > 0:
            pred_columns = [col for col in self.prescriptions.columns[1:] if not col.endswith('_dose_val_rx') and not col.endswith('_dose_unit_rx')]
            active_prescriptions = []
            for col, values in self.prescriptions[pred_columns].items():
                if int(values.values[0]) == 1:
                    active_prescriptions.append(col)
            if len(active_prescriptions) > 0:
                description_yaml['Prescriptions'] = active_prescriptions

        # procedures
        if self.procedures_icd is not None and len(self.procedures_icd.index) > 0:
            new_procedures = []
            for row in self.procedures_icd.iterrows():
                new_procedures.append(row[1].long_title)  # + f" (ICD-{row[1].icd_version}: {row[1].icd_code})"
            description_yaml['Procedures'] = new_procedures

        # description_yaml['Diagnosis'] = self.extracted_diag
        # description_yaml['ICD categories'] = self.icd_categories

        return description_yaml


class Patient_ICU(object):
    def __init__(self, procedureevents=None, outputevents=None, inputevents=None, icustays=None, chartevent_dict=None):
        ## ICU
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.icustays = icustays
        self.chartevents = chartevent_dict
        self.chartevent_categories = ['Cardiovascular', 'Toxicology', 'RoutineVitalSigns', 'NICOM', 'AdmHistory_FHPA', 'Respiratory', 'IABP', 'Dialysis', 'Pulmonary', 'Skin-Assessment',
                                      'Skin-Impairment', 'Skin-Incisions', 'Cardiovascular(Pulses)', 'Neurological', 'Hemodynamics', 'Alarms', 'Pain_Sedation', 'Cardiovascular(PacerData)',
                                      'MDProgressNote', 'GI_GU']
        self.hour_time = None

    def pre_process(self, current_visit_adm):
        # get corresponding admission
        self.adm_time = current_visit_adm.admissions['admittime'].values[0]

    def get_data_until_hour(self, hour_idx, hour_time=None, restrict_to_N_hours=None):
        # Get data for the hour_idx
        if hour_time is None:
            hour_time = self.adm_time + pd.Timedelta(hours=hour_idx)

        # if stay did not start yes, do not include any data
        if hour_time < self.icustays['intime'].iloc[0]:
            self.procedureevents = None
            self.outputevents = None
            self.inputevents = None
            for category in self.chartevents.keys():
                self.chartevents[category] = None
            return hour_time

        self.procedureevents = self.procedureevents[(self.procedureevents.time <= hour_time)] if self.procedureevents is not None else None
        self.outputevents = self.outputevents[(self.outputevents.charttime <= hour_time)] if self.outputevents is not None else None
        self.inputevents = self.inputevents[(self.inputevents.time <= hour_time)] if self.inputevents is not None else None
        if self.chartevents is not None:
            for category, df in self.chartevents.items():
                if df is not None:
                    self.chartevents[category] = df[(df.charttime <= hour_time)]

        if restrict_to_N_hours is not None:
            start_hour_time = hour_time - pd.Timedelta(hours=restrict_to_N_hours)
            self.procedureevents = self.procedureevents[(self.procedureevents.time > start_hour_time)] if self.procedureevents is not None else None
            self.outputevents = self.outputevents[(self.outputevents.charttime > start_hour_time)] if self.outputevents is not None else None
            self.inputevents = self.inputevents[(self.inputevents.time > start_hour_time)] if self.inputevents is not None else None
            if self.chartevents is not None:
                for category, df in self.chartevents.items():
                    if df is not None:
                        self.chartevents[category] = df[(df.charttime > start_hour_time)]

        for row in self.icustays.iterrows():
            if self.icustays.at[row[0], 'outtime'] is not None and hour_time < self.icustays.at[row[0], 'outtime']:
                self.icustays.at[row[0], 'outtime'] = None
                self.icustays.at[row[0], 'los'] = None

    def get_data_at_hour(self, hour_idx):
        # Get data for the hour_idx
        hour_time = self.adm_time + pd.Timedelta(hours=hour_idx)
        hour_in_stay = self.icustays['intime'].values[0] <= hour_time <= self.icustays['outtime'].values[0] # only include data that was measured during the ICU stay to avoid overlap with other ICU stays
        self.procedureevents = self.procedureevents[(self.procedureevents.time == hour_time)] if hour_in_stay and self.procedureevents is not None else None
        self.outputevents = self.outputevents[(self.outputevents.charttime == hour_time)] if hour_in_stay and self.outputevents is not None else None
        self.inputevents = self.inputevents[(self.inputevents.time == hour_time)] if hour_in_stay and self.inputevents is not None else None
        if self.chartevents is not None:
            for category, df in self.chartevents.items():
                if df is not None:
                    self.chartevents[category] = df[(df.charttime == hour_time)] if hour_in_stay else None

        return hour_time

    def get_description(self, hour_time, summary_level=0, out_time=None, icu_los=None, los_only=False):
        description_yaml = {}
        if icu_los is not None: # if LOS is 0, the stay is over, so we don't include it anymore
            description_yaml['LOS'] = f"{icu_los}"
        else: #in validation out_time is always None, so GT LOS is never used (except for ICD prediction at discharge)
            hours_to_discharge = (out_time - hour_time) / pd.Timedelta(hours=1) if out_time is not None else None
            if hours_to_discharge is not None and hours_to_discharge >= 0:
                if random.random() < 0.5:  # delete in 50% of cases to teach model to predict los even if it is not available
                    pass
                else:
                    # add noise and add to description
                    # noise pattern: randomly select hour between -10% to +10% of LOS
                    noised_hours_to_discharge = round(random.uniform(0.8, 1.2) * hours_to_discharge)  #TODO noise activated - for bigger models deactivated
                    description_yaml['LOS'] = f"{int(noised_hours_to_discharge)} hours"

        if los_only:
            return description_yaml

        if self.inputevents is not None and len(self.inputevents.index) > 0:
            description_yaml['Medication'] = {}
            input_cols = [col for col in self.inputevents.columns[1:] if not col.endswith('_rateuom') and not col.endswith('_rate')]
            for col in input_cols:
                values = self.inputevents[['time', col]]
                restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                ts_values = convert_column_to_sorted_periods_string(values, hour_time, time_col_name='time', col=col, check_zero=True, restrict_to_N=restrict_to_N)
                if ts_values != "":
                    description_yaml['Medication'][col] = ts_values
            if len(description_yaml['Medication']) == 0:
                description_yaml.pop('Medication')

        if self.outputevents is not None and len(self.outputevents.index) > 0:
            description_yaml['Output'] = {}
            output_cols = [col for col in self.outputevents.columns[1:] if not col.endswith('_valueuom')]
            for col in output_cols:
                value = self.outputevents[col]
                if not pd.isna(value).all():
                    # iterate over not nan values and how many hours ago they were measured
                    unit = f"({self.outputevents[f'{col}_valueuom'].dropna().values[0]})" if f'{col}_valueuom' in self.outputevents.columns and not pd.isna(
                        self.outputevents[f'{col}_valueuom']).all() else ""
                    restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                    ts_values = convert_column_to_time_events_string(self.outputevents, value, hour_time, category=None, restrict_to_N=restrict_to_N)
                    if ts_values != "":
                        description_yaml['Output'][f"{col}{unit}"] = ts_values
                    else:
                        print(f"Output {col} has no valid values, value: {value}")

        if self.procedureevents is not None and len(self.procedureevents.index) > 0:
            description_yaml['Procedures'] = {}
            procedure_cols = [col for col in self.procedureevents.columns[1:] if not col.endswith('_ordercategoryname') and not col.endswith('_ordercategorydescription')]
            for col in procedure_cols:
                values = self.procedureevents[['time', col]]
                restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                ts_values = convert_column_to_sorted_periods_string(values, hour_time, time_col_name='time', col=col, check_zero=True, restrict_to_N=restrict_to_N)
                if ts_values != "":
                    description_yaml['Procedures'][col] = ts_values
            if len(description_yaml['Procedures']) == 0:
                description_yaml.pop('Procedures')

        if self.chartevents is not None:
            description_yaml["Chart Events"] = {}
            for category in self.chartevent_categories:
                if category in self.chartevents and self.chartevents[category] is not None and len(self.chartevents[category].index) > 0:
                    description_yaml["Chart Events"][category] = {}
                    cols = [col for col in self.chartevents[category].columns[1:] if not col.endswith('_valueuom')]
                    for col in cols:
                        values = self.chartevents[category][col]
                        if not pd.isna(values).all():
                            restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                            ts_values = convert_column_to_time_events_string(self.chartevents, values, hour_time, category=category, restrict_to_N=restrict_to_N)

                            unit = f"({self.chartevents[category][f'{col}_valueuom'].dropna().values[0]})" if f'{col}_valueuom' in self.chartevents[category].columns and not pd.isna(
                                self.chartevents[category][f'{col}_valueuom']).all() else ""
                            pattern_mm = r'(\d+)mm\b'
                            pattern_cm = r'(\d+)cm\b'
                            if re.search(pattern_mm, ts_values):
                                unit = "(mm)"
                                ts_values = re.sub(pattern_mm, r'\1', ts_values)
                            elif re.search(pattern_cm, ts_values):
                                unit = "(cm)"
                                ts_values = re.sub(pattern_cm, r'\1', ts_values)

                            if unit in col:
                                unit = ""

                            if ts_values != "": #can be empty if there is a column with invalid yaml as value (e.g. a dict, two strings for one value)
                                description_yaml["Chart Events"][category][f"{col}{unit}"] = ts_values

            if len(description_yaml["Chart Events"]) == 0:
                description_yaml.pop("Chart Events")

        return description_yaml

    def get_change_description(self, released_from_icu, out_time=None, next_hour_time=None, icu_los=None):
        description_yaml = {}

        if icu_los is not None:
            description_yaml['LOS'] = f"{icu_los}"
        else:
            hours_to_discharge = (out_time - next_hour_time) / pd.Timedelta(hours=1) if out_time is not None else None
            if hours_to_discharge is not None and hours_to_discharge >= 0:
                description_yaml['LOS'] = f"{int(hours_to_discharge)} hours"

        if released_from_icu:
            description_yaml['Disposition'] = "DISCHARGED FROM ICU"

        if self.inputevents is not None and len(self.inputevents.index) > 0:
            input_cols = [col for col in self.inputevents.columns[1:] if not col.endswith('_rateuom') and not col.endswith('_rate')]
            active_medications = []
            for col in input_cols:
                values = self.inputevents[['time', col]]
                if int(values.values[0][1]) == 1:
                    active_medications.append(col)
            if len(active_medications) > 0:
                description_yaml['Medication'] = active_medications

        if self.outputevents is not None and len(self.outputevents.index) > 0:
            description_yaml['Output'] = {}
            output_cols = [col for col in self.outputevents.columns[1:] if not col.endswith('_valueuom')]
            for col in output_cols:
                value = self.outputevents[col]
                if not pd.isna(value).all():
                    # iterate over not nan values and how many hours ago they were measured
                    description_yaml['Output'][f"{col}"] = str(value.values[0])

        if self.procedureevents is not None and len(self.procedureevents.index) > 0:
            all_procedures = []
            procedure_cols = [col for col in self.procedureevents.columns[1:] if not col.endswith('_ordercategoryname') and not col.endswith('_ordercategorydescription')]
            for col in procedure_cols:
                if int(self.procedureevents[col]) == 1:
                    all_procedures.append(col)
            if len(all_procedures) > 0:
                description_yaml['Procedures'] = all_procedures

        if self.chartevents is not None:
            description_yaml["Chart Events"] = {}
            for category in self.chartevent_categories:
                if category in self.chartevents and self.chartevents[category] is not None and len(self.chartevents[category].index) > 0:
                    description_yaml["Chart Events"][category] = {}
                    cols = [col for col in self.chartevents[category].columns[1:] if not col.endswith('_valueuom')]
                    for col in cols:
                        values = self.chartevents[category][col]
                        if not pd.isna(values).all():
                            value = values.values[0]
                            if type(value) == str:
                                pattern_mm = r'(\d+)mm\b'
                                pattern_cm = r'(\d+)cm\b'
                                if re.search(pattern_mm, value):
                                    value = re.sub(pattern_mm, r'\1', value)
                                elif re.search(pattern_cm, value):
                                    value = re.sub(pattern_cm, r'\1', value)

                            description_yaml["Chart Events"][category][f"{col}"] = value
            if len(description_yaml["Chart Events"]) == 0:
                description_yaml.pop("Chart Events")

        return description_yaml

class Patient_ED(object):
    def __init__(self, ed_diagnosis=None, ed_stays=None, ed_medrecon=None, ed_pyxis=None, ed_triage=None, ed_vital=None):
        ## ED
        self.hour_time = None
        self.ed_diagnosis = ed_diagnosis
        self.ed_stays = ed_stays
        self.ed_medrecon = ed_medrecon
        self.ed_pyxis = ed_pyxis
        self.ed_triage = ed_triage
        self.ed_vital = ed_vital
        if self.ed_stays is not None:
            self.intime = self.ed_stays['intime'].values[0]
            self.outtime = self.ed_stays['outtime'].values[0]
            self.stayhours = (self.outtime - self.intime) / pd.Timedelta(hours=1)

    def get_icd_categories(self):
        if self.ed_diagnosis is not None and len(self.ed_diagnosis.index) > 0:
            # get all codes in icd_code where icd_version == 9
            diag_codes = self.ed_diagnosis[self.ed_diagnosis.icd_version == '9'].icd_code.values
            icd_categories = get_icd_category(diag_codes)
            icd_categories = ';'.join(icd_categories)

        else:
            icd_categories = 'unknown'

        return icd_categories

    def pre_process(self):
        if self.ed_vital is not None:
            self.ed_vital = self.ed_vital.sort_values(by=['charttime'])
        self.diag_avail = True
        # self.diagnoses = ';'.join([d.lower() for d in self.ed_diagnosis['icd_title'].unique() if d.lower() != 'unspecified'] if self.ed_diagnosis is not None else ['unspecified'])
        self.icd_categories = self.get_icd_categories()

    def get_data_until_hour(self, hour_idx, hour_time=None, restrict_to_N_hours=None):
        # Get data for the hour_idx
        if hour_time is None:
            hour_time = self.intime + pd.Timedelta(hours=hour_idx)
        self.ed_pyxis = self.ed_pyxis[(self.ed_pyxis.charttime <= hour_time)] if self.ed_pyxis is not None else None
        self.ed_vital = self.ed_vital[(self.ed_vital.charttime <= hour_time)] if self.ed_vital is not None else None

        if restrict_to_N_hours is not None:
            start_hour_time = hour_time - pd.Timedelta(hours=restrict_to_N_hours)
            self.ed_pyxis = self.ed_pyxis[(self.ed_pyxis.charttime > start_hour_time)] if self.ed_pyxis is not None else None
            self.ed_vital = self.ed_vital[(self.ed_vital.charttime > start_hour_time)] if self.ed_vital is not None else None

        if self.outtime is not None and hour_time < self.outtime:  # before end of ED stay or during iterative eval
            self.diag_avail = False
            self.ed_diagnosis = None
            self.ed_stays['outtime'] = None
            self.ed_stays['disposition'] = None
            self.outtime = None
            self.icd_categories = None
            self.stayhours = None

        return hour_time

    def get_data_at_hour(self, hour_idx):
        # Get data for the hour_idx
        hour_time = self.intime + pd.Timedelta(hours=hour_idx)
        self.ed_pyxis = self.ed_pyxis[(self.ed_pyxis.charttime == hour_time)] if self.ed_pyxis is not None else None
        self.ed_vital = self.ed_vital[(self.ed_vital.charttime == hour_time)] if self.ed_vital is not None else None
        return hour_time

    def get_description(self, hour_time, summary_level=0, diag_avail=False, out_time=None, ed_los=None, los_only=False, force_los=False):
        description_yaml = {}
        if ed_los is not None: # if LOS is already known / predicted by model, use prediction
            description_yaml['LOS'] = f"{ed_los}"
        else: # else set LOS to real time until discharge (for training); in validation out_time is always None, so GT LOS is never used (except for ICD prediction at discharge)
            hours_to_discharge = (out_time - hour_time) / pd.Timedelta(hours=1) if out_time is not None else None
            if hours_to_discharge is not None and hours_to_discharge >= 0:
                if not force_los and random.random() < 0.5:  # delete in 50% of cases to teach model to predict los even if it is not available
                    pass
                else:
                    # add noise and add to description
                    # noise pattern: randomly select hour between -10% to +10% of LOS
                    noised_hours_to_discharge = round(random.uniform(0.8, 1.2) * hours_to_discharge)
                    description_yaml['LOS'] = f"{int(noised_hours_to_discharge)} hours"

        if los_only:
            return description_yaml
        description_yaml['General'] = (f"{self.ed_stays['arrival_transport'].values[0].lower()} patient, {'female' if self.ed_stays['gender'].values[0] == 'F' else 'male'}, "
                                       f"{self.ed_stays['race'].values[0].lower()}")
        if self.ed_triage is not None and len(self.ed_triage.index) > 0:
            description_yaml['Chief Complaint'] = self.ed_triage['chiefcomplaint'].values[0].lower() if not pd.isna(self.ed_triage['chiefcomplaint'].values[0]) else "unknown"
            # delete nan columns from self.ed_triage
            description_yaml['Admission Vitals'] = {}
            for col, unit in zip(['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'pain'], ['F', 'bpm', 'breaths/min', '%', '', '/10']):
                value = self.ed_triage[col].values[0]
                if not pd.isna(value):
                    try:
                        description_yaml['Admission Vitals'][col] = (f"{int(self.ed_triage[col].values[0])}{unit}")
                    except:
                        description_yaml['Admission Vitals'][col] = (f"{self.ed_triage[col].values[0]}{unit}")

        if self.ed_medrecon is not None and len(self.ed_medrecon.index) > 0:
            description_yaml['Medicine Reconciliation'] = ', '.join(self.ed_medrecon['name'].unique())

        if self.ed_vital is not None and len(self.ed_vital.index) > 0:
            description_yaml['Vital Measurements'] = {}
            for col, unit in zip(['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'rhythm', 'pain'], [' (F)', ' (bpm)', ' (breaths/min)', ' (%)', '', '', '', ' (1-10)']):
                if col in self.ed_vital.columns:
                    value = self.ed_vital[col]
                    if not pd.isna(value).all():
                        # iterate over not nan values and how many hours ago they were measured
                        restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                        ts_values = convert_column_to_time_events_string(self.ed_vital, value, hour_time, category=None, restrict_to_N=restrict_to_N)
                        description_yaml['Vital Measurements'][f"{col}{unit}"] = ts_values

        if self.ed_pyxis is not None and len(self.ed_pyxis.index) > 0:
            description_yaml['Medication'] = {}
            for col, values in self.ed_pyxis.items():
                if col != "charttime" and len(values.dropna()) > 0:
                    values = self.ed_pyxis[['charttime', col]]
                    restrict_to_N = 5 if summary_level == 2 else 1 if summary_level >= 3 else None
                    ts_values = convert_column_to_sorted_periods_string(values, hour_time, time_col_name='charttime', col=col, restrict_to_N=restrict_to_N)
                    description_yaml['Medication'][col] = ts_values

        if diag_avail:
            description_yaml['Diagnosis'] = ', '.join([d.lower() for d in self.ed_diagnosis['icd_title'].unique() if d.lower() != 'unspecified'] if self.ed_diagnosis is not None else [])

        return description_yaml

    def get_change_description(self, out_time=None, next_hour_time=None, ed_los=None):
        description_yaml = {}

        if ed_los is not None:
            description_yaml['LOS'] = f"{ed_los}"
        else:
            hours_to_discharge = (out_time - next_hour_time) / pd.Timedelta(hours=1) if out_time is not None else None
            if hours_to_discharge is not None and hours_to_discharge >= 0:
                description_yaml['LOS'] = f"{int(hours_to_discharge)} hours"

        if self.ed_vital is not None and len(self.ed_vital.index) > 0:
            description_yaml['Vital Measurements'] = {}
            for col, unit in zip(['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'rhythm', 'pain'], [' (F)', ' (bpm)', ' (breaths/min)', ' (%)', '', '', '', '(1-10)']):
                value = self.ed_vital[col]
                if not pd.isna(value).all():
                    description_yaml['Vital Measurements'][f"{col}"] = str(value.values[0])

        if self.ed_pyxis is not None and len(self.ed_pyxis.index) > 0:
            given_medications = []
            for col, values in self.ed_pyxis.items():
                if col != "charttime":
                    given_medications.append(col)

            if len(given_medications) > 0:
                description_yaml['Medication'] = given_medications

        # description_yaml['Diagnosis'] = self.diagnoses
        # description_yaml['ICD categories'] = self.icd_categories

        return description_yaml


class Patient_Visit(object):
    def __init__(self, patient_adm=None, patient_icu=None, patient_ed=None):
        self.patient_ed = patient_ed
        self.patient_adm = patient_adm
        self.patient_icu = patient_icu


def get_patient_admission(key_subject_id, key_hadm_id, dfs, df_base_core):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #
    #   NOTES: Identifiers which specify the patient. More information about
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # -> FILTER data
    ##-> HOSP
    # Filter dataframes based on the key_subject_id and key_hadm_id
    f_df_base_core = df_base_core[(df_base_core.subject_id == key_subject_id) & (df_base_core.hadm_id == key_hadm_id)]
    f_df_omr = dfs['df_omr'][(dfs['df_omr'].subject_id == key_subject_id)]
    f_df_admissions = dfs['df_admissions'][(dfs['df_admissions'].subject_id == key_subject_id) & (dfs['df_admissions'].hadm_id == key_hadm_id)]
    f_df_patients = dfs['df_patients'][(dfs['df_patients'].subject_id == key_subject_id)]
    f_df_transfers = dfs['df_transfers'][(dfs['df_transfers'].subject_id == key_subject_id) & (dfs['df_transfers'].hadm_id == key_hadm_id)]

    # Merge data into single patient structure
    f_df_core = f_df_base_core
    f_df_core = f_df_core.merge(f_df_admissions, how='left')
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core = f_df_core.merge(f_df_transfers, how='left')

    f_df_diagnoses_icd = dfs['df_diagnoses_icd'][(dfs['df_diagnoses_icd'].subject_id == key_subject_id) & (dfs['df_diagnoses_icd'].hadm_id == key_hadm_id)]
    f_df_drgcodes = dfs['df_drgcodes'][(dfs['df_drgcodes'].subject_id == key_subject_id) & (dfs['df_drgcodes'].hadm_id == key_hadm_id)]
    f_df_labevents = dfs['df_labevents'][(dfs['df_labevents'].subject_id == key_subject_id) & (dfs['df_labevents'].hadm_id == key_hadm_id)]
    f_df_microbiologyevents = dfs['df_microbiologyevents'][
        (dfs['df_microbiologyevents'].subject_id == key_subject_id) & (dfs['df_microbiologyevents'].hadm_id == key_hadm_id)]
    f_df_prescriptions = dfs['df_prescriptions'][
        (dfs['df_prescriptions'].subject_id == key_subject_id) & (dfs['df_prescriptions'].hadm_id == key_hadm_id)]
    f_df_procedures_icd = dfs['df_procedures_icd'][
        (dfs['df_procedures_icd'].subject_id == key_subject_id) & (dfs['df_procedures_icd'].hadm_id == key_hadm_id)]
    f_df_services = dfs['df_services'][(dfs['df_services'].subject_id == key_subject_id) & (dfs['df_services'].hadm_id == key_hadm_id)]

    # Merge content from dictionaries for notes
    f_df_disch_notes = dfs['df_disch_notes'][(dfs['df_disch_notes'].subject_id == key_subject_id) & (dfs['df_disch_notes'].hadm_id == key_hadm_id)]
    f_df_rad_notes = dfs['df_rad_notes'][(dfs['df_rad_notes'].subject_id == key_subject_id) & (dfs['df_rad_notes'].hadm_id == key_hadm_id)]
    rad_node_ids = f_df_rad_notes['note_id']
    f_df_rad_details = dfs['df_rad_details'][(dfs['df_rad_details'].subject_id == key_subject_id)]
    f_df_rad_details = f_df_rad_details[f_df_rad_details['note_id'].isin(rad_node_ids)]

    # Process rad_notes
    exam_name_details = f_df_rad_details[f_df_rad_details['field_name'] == 'exam_name']
    exam_name_aggregated = exam_name_details.groupby('note_id')['field_value'].apply(lambda x: '; '.join(x)).reset_index()
    exam_name_aggregated = exam_name_aggregated.rename(columns={'field_value': 'exam_name'})
    f_df_rad_notes = f_df_rad_notes.merge(exam_name_aggregated, on='note_id', how='left')

    exam_code_details = f_df_rad_details[f_df_rad_details['field_name'] == 'exam_code']
    exam_code_aggregated = exam_code_details.groupby('note_id')['field_value'].apply(lambda x: '; '.join(x)).reset_index()
    exam_code_aggregated = exam_code_aggregated.rename(columns={'field_value': 'exam_code'})
    f_df_rad_notes = f_df_rad_notes.merge(exam_code_aggregated, on='note_id', how='left')

    has_addendum = f_df_rad_details[f_df_rad_details['field_name'] == 'addendum_note_id']
    if not has_addendum.empty:
        has_addendum = has_addendum.rename(columns={'field_value': 'addendum_note_id'})
        for _, row in has_addendum.iterrows():
            note_id = row['note_id']
            addendum_id = row['addendum_note_id']
            try:
                addendum_text = f_df_rad_notes[f_df_rad_notes['note_id'] == addendum_id]['text'].item()
            except Exception as e:
                # addendum missing in rad_notes (2x in dataset) -> add empty/no addendum
                addendum_text = ""
            f_df_rad_notes.loc[f_df_rad_notes['note_id'] == note_id, 'text'] += addendum_text
        f_df_rad_notes = f_df_rad_notes[~f_df_rad_notes['note_id'].isin(has_addendum['addendum_note_id'])]
    f_df_rad_notes = f_df_rad_notes.drop_duplicates(subset='text')

    # Create & Populate patient structure
    omr = convert_ts_to_columns(f_df_omr, label_col_name='result_name', value_col_name='result_value', time_col_name=['chartdate', 'seq_num'],
                                aggfunc='last')
    admissions = f_df_admissions
    patients = f_df_patients
    transfers = create_hourly_event_df(f_df_transfers, label_col_name='careunit', start_time_col="intime", end_time_col="outtime",
                                       time_col_name='time')
    services = f_df_services.sort_values(by='transfertime')
    services['endtime'] = services['transfertime'].shift(-1)
    services['endtime'].iloc[-1] = admissions['dischtime'].values[0]
    services = create_hourly_event_df(services, label_col_name='curr_service', start_time_col="transfertime", end_time_col="endtime",
                                      time_col_name='time')

    core = f_df_core

    # HOSP section
    diagnoses_icd = f_df_diagnoses_icd
    drgcodes = f_df_drgcodes
    labevents = convert_ts_to_columns(f_df_labevents, label_col_name='label',
                                      info_cols=['valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments'],
                                      value_col_name='valuenum',
                                      secondary_value_col='value', third_value_col='comments', time_col_name='charttime',
                                      aggfunc=mean_or_last_agg)

    microbiologyevents = f_df_microbiologyevents
    if not microbiologyevents.empty:
        microbiologyevents = convert_microbiology(microbiologyevents)
    prescriptions = create_hourly_event_df(f_df_prescriptions, label_col_name='nonproprietaryname', start_time_col="starttime",
                                           end_time_col="stoptime", time_col_name='time',
                                           info_cols=['dose_val_rx', 'dose_unit_rx'], disch_time= admissions.dischtime.iloc[0], admit_time=admissions.admittime.iloc[0])
    procedures_icd = f_df_procedures_icd

    # Notes section
    dsnotes = f_df_disch_notes
    radnotes = f_df_rad_notes

    # Set empty dataframes to None
    if len(omr.index) == 0:
        omr = None
    if len(transfers.index) == 0:
        transfers = None
    if len(services.index) == 0:
        services = None
    if len(diagnoses_icd.index) == 0:
        diagnoses_icd = None
    if len(drgcodes.index) == 0:
        drgcodes = None
    if len(labevents.index) == 0:
        labevents = None
    if len(microbiologyevents.index) == 0:
        microbiologyevents = None
    if len(prescriptions.index) == 0:
        prescriptions = None
    if len(procedures_icd.index) == 0:
        procedures_icd = None
    if len(dsnotes.index) == 0:
        dsnotes = None
    if len(radnotes.index) == 0:
        radnotes = None

    # Create patient object and return
    Patient_Admission = Patient_ADM(omr, admissions, patients, transfers, core, diagnoses_icd, drgcodes, labevents, microbiologyevents, prescriptions,
                                    procedures_icd, services, dsnotes, radnotes)

    return Patient_Admission


# GET FULL MIMIC IV PATIENT RECORD USING DATABASE KEYS
def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id, dfs):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #
    #   NOTES: Identifiers which specify the patient. More information about
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # -> FILTER data
    ##-> CORE
    # Filter dataframes based on the key_subject_id, key_hadm_id, and key_stay_id
    f_df_procedureevents = dfs['df_procedureevents'][
        (dfs['df_procedureevents'].subject_id == key_subject_id) &
        (dfs['df_procedureevents'].hadm_id == key_hadm_id) &
        (dfs['df_procedureevents'].stay_id == key_stay_id)
        ]
    f_df_outputevents = dfs['df_outputevents'][
        (dfs['df_outputevents'].subject_id == key_subject_id) &
        (dfs['df_outputevents'].hadm_id == key_hadm_id) &
        (dfs['df_outputevents'].stay_id == key_stay_id)
        ]
    f_df_inputevents = dfs['df_inputevents'][
        (dfs['df_inputevents'].subject_id == key_subject_id) &
        (dfs['df_inputevents'].hadm_id == key_hadm_id) &
        (dfs['df_inputevents'].stay_id == key_stay_id)
        ]
    f_df_icustays = dfs['df_icustays'][
        (dfs['df_icustays'].subject_id == key_subject_id) &
        (dfs['df_icustays'].hadm_id == key_hadm_id) &
        (dfs['df_icustays'].stay_id == key_stay_id)
        ]
    f_df_chartevents = dfs['df_chartevents'][
        (dfs['df_chartevents'].subject_id == key_subject_id) &
        (dfs['df_chartevents'].hadm_id == key_hadm_id) &
        (dfs['df_chartevents'].stay_id == key_stay_id)
        ]

    # ICU
    procedureevents = f_df_procedureevents
    procedure_categories = ['Continuous Procedures', 'Dialysis', 'Imaging', 'Intubation/Extubation', 'Invasive Lines', 'Procedures',
                            'Significant Events', 'Tubes', 'Ventilation']
    procedureevents = procedureevents[procedureevents['ordercategoryname'].isin(procedure_categories)]
    procedureevents = create_hourly_event_df(procedureevents, label_col_name='label', start_time_col="starttime", end_time_col="endtime",
                                             time_col_name='time',
                                             info_cols=['ordercategoryname', 'ordercategorydescription'])

    outputevents = convert_ts_to_columns(f_df_outputevents, label_col_name='label', value_col_name='value', time_col_name='charttime',
                                         info_cols=['valueuom'], aggfunc='sum')

    inputevents = create_hourly_event_df(f_df_inputevents, label_col_name='label', start_time_col="starttime", end_time_col="endtime",
                                         time_col_name='time', info_cols=['rate', 'rateuom'])

    icustays = f_df_icustays

    chartevents = f_df_chartevents
    categories = ['Cardiovascular', 'Toxicology', 'Routine Vital Signs', 'NICOM', 'Adm History/FHPA', 'Respiratory', 'IABP', 'Dialysis', 'Pulmonary',
                  'Skin - Assessment', 'Skin - Impairment',
                  'Skin - Incisions', 'Cardiovascular (Pulses)', 'Neurological', 'Hemodynamics', 'Alarms', 'Pain/Sedation',
                  'Cardiovascular (Pacer Data)', 'MD Progress Note', 'GI/GU']
    chartevent_dict = {}
    for category in categories:
        chartevents_category = chartevents[chartevents['category'] == category]
        if not chartevents_category.empty:
            chartevents_category = convert_ts_to_columns(chartevents_category, label_col_name='label', value_col_name='value',
                                                         time_col_name='charttime', info_cols=['valueuom'],
                                                         aggfunc=mean_or_last_agg)
        else:
            chartevents_category = None

        chartevent_dict[category.replace('/', '_').replace(' ', '')] = chartevents_category

    # Set empty dataframes to None
    if len(procedureevents.index) == 0:
        procedureevents = None
    if len(outputevents.index) == 0:
        outputevents = None
    if len(inputevents.index) == 0:
        inputevents = None
    # if len(datetimeevents.index) == 0:
    #     datetimeevents = None

    # Create patient object and return
    Patient_ICUstay = Patient_ICU(procedureevents, outputevents, inputevents, icustays, chartevent_dict)

    return Patient_ICUstay


def get_patient_edstay(key_subject_id, key_stay_id, dfs):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #
    #   NOTES: Identifiers which specify the patient. More information about
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # -> FILTER data
    ##-> CORE
    # Filter dataframes based on the key_subject_id and key_stay_id
    f_df_ed_diagnosis = dfs['df_ed_diagnosis'][
        (dfs['df_ed_diagnosis'].subject_id == key_subject_id) &
        (dfs['df_ed_diagnosis'].stay_id == key_stay_id)
        ]
    f_df_ed_stays = dfs['df_ed_stays'][
        (dfs['df_ed_stays'].subject_id == key_subject_id) &
        (dfs['df_ed_stays'].stay_id == key_stay_id)
        ]
    f_df_ed_medrecon = dfs['df_ed_medrecon'][
        (dfs['df_ed_medrecon'].subject_id == key_subject_id) &
        (dfs['df_ed_medrecon'].stay_id == key_stay_id)
        ]
    f_df_ed_pyxis = dfs['df_ed_pyxis'][
        (dfs['df_ed_pyxis'].subject_id == key_subject_id) &
        (dfs['df_ed_pyxis'].stay_id == key_stay_id)
        ]
    f_df_ed_triage = dfs['df_ed_triage'][
        (dfs['df_ed_triage'].subject_id == key_subject_id) &
        (dfs['df_ed_triage'].stay_id == key_stay_id)
        ]
    f_df_ed_vital = dfs['df_ed_vital'][
        (dfs['df_ed_vital'].subject_id == key_subject_id) &
        (dfs['df_ed_vital'].stay_id == key_stay_id)
        ]

    # Aggregate gsn_rn and gsn values in pyxis for same medication in same hour
    # convert category to string so groupby works
    f_df_ed_pyxis['name'] = f_df_ed_pyxis['name'].astype(str)
    f_df_ed_pyxis = f_df_ed_pyxis.groupby(['subject_id', 'stay_id', 'charttime', 'med_rn', 'name']).agg({
        'gsn_rn': lambda x: ';'.join(x.astype(str)),
        'gsn': lambda x: ';'.join(x.astype(str))
    }).reset_index()

    ed_diagnosis = f_df_ed_diagnosis
    ed_stays = f_df_ed_stays
    ed_medrecon = f_df_ed_medrecon
    ed_pyxis = f_df_ed_pyxis
    ed_pyxis['mock_value'] = 1
    ed_pyxis = convert_ts_to_columns(ed_pyxis, label_col_name='name', value_col_name='mock_value', time_col_name='charttime')
    ed_triage = f_df_ed_triage
    ed_vital = f_df_ed_vital
    ed_vital = ed_vital.groupby(['charttime']).agg(mean_or_last_agg).reset_index()

    if len(ed_diagnosis.index) == 0:
        ed_diagnosis = None
    if len(ed_medrecon.index) == 0:
        ed_medrecon = None
    if len(ed_pyxis.index) == 0:
        ed_pyxis = None
    if len(ed_triage.index) == 0:
        ed_triage = None
    if len(ed_vital.index) == 0:
        ed_vital = None

    # Create patient object and return
    Patient_EDstay = Patient_ED(ed_diagnosis, ed_stays, ed_medrecon, ed_pyxis, ed_triage, ed_vital)

    return Patient_EDstay


def split_dataframes_for_workers(dfs, patient_ids, num_workers):
    """
    Pre-slice the dataframes based on patient_ids and distribute them among workers.
    """
    # Split patient_ids evenly among workers
    patient_id_splits = np.array_split(sorted(list(set(patient_ids))), num_workers)

    worker_data = []
    for patient_subset in patient_id_splits:
        worker_dfs = {name: df[df['subject_id'].isin(patient_subset)].copy() if 'subject_id' in df.keys() else df.copy() for name, df in dfs.items()}
        worker_data.append((patient_subset, worker_dfs))

    # remove the original dfs to save memory
    for key in list(dfs.keys()):
        del dfs[key]
    # garbage collect
    gc.collect()
    return worker_data

def worker_function(worker_input, df_ids, df_icu_ids, df_ed_ids, df_base_core, mimiciv_path):
    """
    Process a subset of patients for a single worker using its own subset of DataFrames.
    """
    patient_ids, dfs = worker_input  # Unpack the tuple received from imap

    for patient_id in tqdm(patient_ids, desc="Processing patients", leave=False):
        # Generate patient data and update the dictionary
        generate_patient_helper_greedy(patient_id, dfs, df_ids, df_icu_ids, df_ed_ids, df_base_core, mimiciv_path)


def generate_patient_helper_greedy(patient_id, dfs, df_ids, df_icu_ids, df_ed_ids, df_base_core, mimiciv_path):
    try:
        # collect meta information: "adm_id", "icu_stay_ids", "ed_stay_id", "stay_type", "start_time", "end_time", "hours"
        patient_folder = mimiciv_path + 'all_admissions/' + f"patient_{patient_id}/"

        if os.path.exists(patient_folder + 'stay_to_ids_dict.json'): # patient already processed
            print(f"Patient {patient_id} already processed")
            return

        # create patient folder
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)

        df_icustays = dfs['df_icustays']
        df_ed_stays = dfs['df_ed_stays']

        meta_info = []
        stay_to_ids_dict = {}

        patient_dict = {'ed_stay_ids': [], 'admissions': {}}
        # Extract information for patient
        key_hadm_ids = df_ids[df_ids.subject_id == patient_id].hadm_id

        for adm_id in key_hadm_ids:
            patient_dict['admissions'][adm_id] = {}
            # get all icu stays for patient
            icu_stay_ids = df_icu_ids[(df_icu_ids.subject_id == patient_id) & (df_icu_ids.hadm_id == adm_id)].stay_id
            patient_dict['admissions'][adm_id]['icu_stay_ids'] = icu_stay_ids.tolist()

        # get all ed stays for patient
        ed_stay_ids = df_ed_ids[(df_ed_ids.subject_id == patient_id)].stay_id
        patient_dict['ed_stay_ids'] = ed_stay_ids.tolist()

        for adm_id in patient_dict['admissions'].keys():
            # add admission as row to meta_df as type "admission", icu and ed stay ids are empty
            adm_start_time = df_base_core[(df_base_core.subject_id == patient_id) & (df_base_core.hadm_id == adm_id)].admittime.values[0]
            adm_end_time = df_base_core[(df_base_core.subject_id == patient_id) & (df_base_core.hadm_id == adm_id)].dischtime.values[0]
            stayhours = (adm_end_time - adm_start_time) / pd.Timedelta(hours=1)
            meta_info.append({"adm_id": adm_id, "icu_stay_id": np.nan, "ed_stay_id": np.nan, "stay_type": "admission", "start_time": adm_start_time,
                              "end_time": adm_end_time, "hours": stayhours})

            # linked ICU stays
            for key_stay_id in patient_dict['admissions'][adm_id]['icu_stay_ids']:
                icustays = df_icustays[
                    (df_icustays.subject_id == patient_id) & (df_icustays.hadm_id == adm_id) & (df_icustays.stay_id == key_stay_id)]

                # add icu stay as row to meta_df as type "icu", ed stay id is empty
                icu_start_time = icustays[(icustays.subject_id == patient_id) & (icustays.hadm_id == adm_id) & (icustays.stay_id == key_stay_id)].intime.values[0]
                icu_end_time = icustays[(icustays.subject_id == patient_id) & (icustays.hadm_id == adm_id) & (icustays.stay_id == key_stay_id)].outtime.values[0]
                stayhours = (icu_end_time - icu_start_time) / pd.Timedelta(hours=1)
                meta_info.append(
                    {"adm_id": adm_id, "icu_stay_id": key_stay_id, "ed_stay_id": np.nan, "stay_type": "icu", "start_time": icu_start_time,
                     "end_time": icu_end_time, "hours": stayhours})

        # Unlinked emergency department stays - can be linked to an admission (if patient is admitted after ED stay, but do not have to)
        for key_stay_id in patient_dict['ed_stay_ids']:
            ed_patient_stay = get_patient_edstay(patient_id, key_stay_id, dfs)
            ed_stays = df_ed_stays[(df_ed_stays.subject_id == patient_id) & (df_ed_stays.stay_id == key_stay_id)]

            # add ed stay as row to meta_df as type "ed", icu stay id is empty
            ed_start_time = ed_stays[(ed_patient_stay.ed_stays.subject_id == patient_id) & (ed_stays.stay_id == key_stay_id)].intime.values[0]
            ed_end_time = ed_stays[(ed_patient_stay.ed_stays.subject_id == patient_id) & (ed_stays.stay_id == key_stay_id)].outtime.values[0]
            stayhours = (ed_end_time - ed_start_time) / pd.Timedelta(hours=1)
            # check if ed stay is linked to an admission
            if ed_patient_stay.ed_stays[(ed_stays.subject_id == patient_id) & (ed_stays.stay_id == key_stay_id)].hadm_id.isnull().values[0]:
                adm_id = np.nan
            else:
                adm_id = ed_patient_stay.ed_stays[(ed_stays.subject_id == patient_id) & (ed_stays.stay_id == key_stay_id)].hadm_id.values[0]
            meta_info.append({"adm_id": adm_id, "icu_stay_id": np.nan, "ed_stay_id": key_stay_id, "stay_type": "ed", "start_time": ed_start_time,
                              "end_time": ed_end_time, "hours": stayhours})

        # save meta_info as csv
        meta_df = pd.DataFrame(meta_info, columns=["adm_id", "icu_stay_id", "ed_stay_id", "stay_type", "start_time", "end_time", "hours"])
        # sort by start time
        meta_df = meta_df.sort_values(by=['start_time'])
        meta_df.to_csv(patient_folder + "meta_info.csv", index=False)

        # save complete stay identifiers
        # prior_stays = []
        for idx, meta_row in meta_df.iterrows():
            if meta_row['stay_type'] == 'admission':
                curr_stay_id = f"{patient_id}_{meta_row['adm_id']}"

                # check if any ICU visits are linked to the current admission in the meta_data
                linked_icu_rows = meta_df[(meta_df['adm_id'] == meta_row['adm_id']) & (meta_df['stay_type'] == 'icu')]
                linked_icu_ids = list(linked_icu_rows['icu_stay_id'].values)

                # check if any ED visits are linked to the current admission in the meta_data
                linked_ed_rows = meta_df[(meta_df['adm_id'] == meta_row['adm_id']) & (meta_df['stay_type'] == 'ed')]
                # keep ed stay with later start time
                if len(linked_ed_rows) > 1:
                    linked_ed_rows = linked_ed_rows.sort_values(by=['start_time'], ascending=False)
                    print("WARNING: dropping old ED stay due to multiple linked ED stays")

                linked_ed_id = linked_ed_rows['ed_stay_id'].values[0] if not linked_ed_rows.empty else None

                stay_to_ids_dict[curr_stay_id] = {'patient_id': int(patient_id), 'adm_id': meta_row['adm_id'], 'icu_stay_ids': linked_icu_ids, 'ed_stay_id': linked_ed_id,
                                                  "ed_stay_hours": linked_ed_rows['hours'].values[0] if not linked_ed_rows.empty else 0,
                                                  "hosp_stay_hours": meta_row['hours']}  # , "prior_stays": prior_stays.copy()

                patient_adm = get_patient_admission(patient_id, meta_row['adm_id'], dfs, df_base_core)
                if len(linked_icu_ids) > 0:
                    icu_stays = {}
                    for icu_id in linked_icu_ids:
                        patient_icu = get_patient_icustay(patient_id, meta_row['adm_id'], icu_id, dfs)
                        icu_stays[icu_id] = patient_icu
                else:
                    icu_stays = None
                if len(linked_ed_rows) > 0:
                    patient_ed = get_patient_edstay(patient_id, linked_ed_id, dfs)
                else:
                    patient_ed = None

                full_patient = Patient_Visit(patient_adm, icu_stays, patient_ed)
                with gzip.open(patient_folder + f'patient_stay_{curr_stay_id}.pkl.gz', 'wb') as f:
                    pickle.dump(full_patient, f, protocol=pickle.HIGHEST_PROTOCOL)

                # prior_stays.append(curr_stay_id)

            elif meta_row['stay_type'] == 'ed' and np.isnan(meta_row['adm_id']):  # ED stays without admission
                curr_stay_id = f"{patient_id}_{meta_row['ed_stay_id']}"
                stay_to_ids_dict[curr_stay_id] = {'patient_id': int(patient_id), 'adm_id': None, 'icu_stay_ids': [], 'ed_stay_id': meta_row['ed_stay_id'], "ed_stay_hours": meta_row['hours'],
                                                  "hosp_stay_hours": None}  # , "prior_stays": prior_stays.copy()
                # prior_stays.append(curr_stay_id)

                patient_ed = get_patient_edstay(patient_id, meta_row['ed_stay_id'], dfs)
                full_patient = Patient_Visit(None, None, patient_ed)
                with gzip.open(patient_folder + f'patient_stay_{curr_stay_id}.pkl.gz', 'wb') as f:
                    pickle.dump(full_patient, f, protocol=pickle.HIGHEST_PROTOCOL)

            else:  # ICU stay or linked ED stay, so not a separate patient model
                continue

        # save cxr data
        # Access the required dataframes from the dictionary
        f_df_mimic_cxr_split = dfs['df_mimic_cxr_split'][(dfs['df_mimic_cxr_split'].subject_id == patient_id)]
        f_df_mimic_cxr_chexpert = dfs['df_mimic_cxr_chexpert'][(dfs['df_mimic_cxr_chexpert'].subject_id == patient_id)]
        f_df_mimic_cxr_metadata = dfs['df_mimic_cxr_metadata'][(dfs['df_mimic_cxr_metadata'].subject_id == patient_id)]
        f_df_mimic_cxr_negbio = dfs['df_mimic_cxr_negbio'][(dfs['df_mimic_cxr_negbio'].subject_id == patient_id)]

        # Merge data into a single patient structure
        if len(f_df_mimic_cxr_split.index) > 0:
            f_df_cxr = f_df_mimic_cxr_split
            f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_chexpert, how='left')
            f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_metadata, how='left')
            f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_negbio, how='left')
            # Save the result to a CSV file
            f_df_cxr.to_csv(patient_folder + 'cxr.csv.gz', compression='gzip', index=False)
        else:
            f_df_cxr = pd.DataFrame(columns=f_df_mimic_cxr_split.columns)
            f_df_cxr.to_csv(patient_folder + 'cxr.csv.gz', compression='gzip', index=False)

        # save stay_to_ids_dict
        with open(patient_folder + 'stay_to_ids_dict.json', 'w') as f:
            json.dump(stay_to_ids_dict, f)
        # trigger garbage collection
        gc.collect()

    except Exception as e:
        print(f"ERROR: {e}, patient {patient_id}")
        return None  # Or handle the error appropriately


def generate_all_mimiciv_patient_objects(df_ids, df_icu_ids, df_ed_ids, mimiciv_path, dfs, df_base_core):
    df_patient_ids = df_ids[['subject_id']].drop_duplicates()
    df_ed_ids = df_ed_ids[['subject_id', 'stay_id']].drop_duplicates()
    patient_ids = np.unique(np.concatenate([df_patient_ids.subject_id, df_ed_ids.subject_id]))

    dfs['df_diagnoses_icd'] = dfs['df_diagnoses_icd'].merge(dfs['df_d_icd_diagnoses'], how='left')
    dfs['df_procedures_icd'] = dfs['df_procedures_icd'].merge(dfs['df_d_icd_procedures'], how='left')
    # choose relevant columns
    dfs['df_labevents'] = dfs['df_labevents'].merge(dfs['df_d_labitems'], how='left')

    dfs['df_procedureevents'] = dfs['df_procedureevents'].merge(dfs['df_d_items'], how='left')
    dfs['df_outputevents'] = dfs['df_outputevents'].merge(dfs['df_d_items'], how='left')
    dfs['df_inputevents'] = dfs['df_inputevents'].merge(dfs['df_d_items'], how='left')
    dfs['df_chartevents'] = dfs['df_chartevents'].merge(dfs['df_d_items'], how='left')

    # drop unnecessary dfs (already merged)
    _ = dfs.pop('df_d_items')
    _ = dfs.pop('df_d_icd_diagnoses')
    _ = dfs.pop('df_d_icd_procedures')
    _ = dfs.pop('df_d_labitems')

    n_workers = 128
    starttime = time.time()
    worker_data = split_dataframes_for_workers(dfs, patient_ids, n_workers)
    print("Time to split data into patient-specific data: ", time.time() - starttime)
    # Prepare the worker function with partial
    start = time.time()
    worker_func = partial(worker_function, df_ids=df_ids, df_icu_ids=df_icu_ids, df_ed_ids=df_ed_ids,
                          df_base_core=df_base_core, mimiciv_path=mimiciv_path)
    # Use multiprocessing pool to process each worker's data
    with Pool(processes=n_workers) as pool:
        list(tqdm(pool.imap(worker_func, worker_data), total=n_workers, desc="Overall progress"))
    print("Time to process all patients: ", time.time() - start)


    # iterative approach for debugging, iterate over worker_data
    # for worker_input in worker_data:
    #     worker_function(worker_input, df_ids, df_icu_ids, df_ed_ids, df_base_core, mimiciv_path)

    # merge saved stay_to_ids_dicts
    stay_to_ids_dict = {}
    # iterate folders in mimiciv_path, 'all_admissions'
    for patient_folder in tqdm(os.listdir(os.path.join(mimiciv_path, 'all_admissions'))):
        with open(os.path.join(mimiciv_path, 'all_admissions', patient_folder, 'stay_to_ids_dict.json'), 'r') as f:
            stays_dict = json.load(f)
        stay_to_ids_dict.update(stays_dict)
    print("Time to merge stay_to_ids_dicts: ", time.time() - start)
    with open(f"{mimiciv_path}/stay_to_ids_dict.json", "w") as outfile:
        json.dump(stay_to_ids_dict, outfile)

    return stay_to_ids_dict


def main():
    '''
    Load MIMIC IV data into DataFrames
    '''

    start = time.time()
    (df_omr, df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_drgcodes, df_labevents,
     df_microbiologyevents, df_prescriptions, df_procedures_icd, df_services, df_d_icd_diagnoses, df_d_icd_procedures,
     df_d_hcpcs, df_d_labitems, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_chartevents, df_d_items, df_mimic_cxr_split,
     df_mimic_cxr_chexpert, df_mimic_cxr_metadata, df_mimic_cxr_negbio, df_disch_notes, df_rad_notes, df_rad_details,
     df_ed_diagnosis, df_ed_stays, df_ed_medrecon, df_ed_pyxis, df_ed_triage, df_ed_vital) = load_mimiciv(mimiciv_path, mimiciv_imgcxr_path,
                                                                                                          mimiciv_note_path, mimiciv_ed_path)
    print("Time to load MIMIC-IV data: ", time.time() - start)

    # save as dictionary for easier access
    dfs = {'df_omr': df_omr, 'df_admissions': df_admissions, 'df_patients': df_patients, 'df_transfers': df_transfers, 'df_diagnoses_icd': df_diagnoses_icd,
           'df_drgcodes': df_drgcodes, 'df_labevents': df_labevents, 'df_microbiologyevents': df_microbiologyevents, 'df_prescriptions': df_prescriptions,
           'df_procedures_icd': df_procedures_icd, 'df_services': df_services, 'df_d_icd_diagnoses': df_d_icd_diagnoses, 'df_d_icd_procedures': df_d_icd_procedures,
           'df_d_hcpcs': df_d_hcpcs, 'df_d_labitems': df_d_labitems, 'df_procedureevents': df_procedureevents, 'df_outputevents': df_outputevents,
           'df_inputevents': df_inputevents, 'df_icustays': df_icustays, 'df_chartevents': df_chartevents, 'df_d_items': df_d_items, 'df_mimic_cxr_split': df_mimic_cxr_split,
           'df_mimic_cxr_chexpert': df_mimic_cxr_chexpert, 'df_mimic_cxr_metadata': df_mimic_cxr_metadata, 'df_mimic_cxr_negbio': df_mimic_cxr_negbio, 'df_disch_notes': df_disch_notes,
           'df_rad_notes': df_rad_notes, 'df_rad_details': df_rad_details, 'df_ed_diagnosis': df_ed_diagnosis, 'df_ed_stays': df_ed_stays,
           'df_ed_medrecon': df_ed_medrecon, 'df_ed_pyxis': df_ed_pyxis, 'df_ed_triage': df_ed_triage, 'df_ed_vital': df_ed_vital}

    # -> MASTER DICTIONARY of health items
    # Generate dictionary for chartevents, labevents and HCPCS
    # Get Chartevent items with labels & category

    # ## -> GET LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV (subject_id, hadm_id, stay_id)
    df_base_core = df_admissions.merge(df_patients, how='left').merge(df_transfers, how='left')

    # Generate integer representations of categorical variables in core
    core_var_select_list = ['gender', 'race', 'marital_status', 'language', 'insurance']  # ethnicity changed to "race"
    # core_var_select_int_list = ['gender_int', 'race_int', 'marital_status_int', 'language_int', 'insurance_int']
    df_base_core[core_var_select_list] = df_base_core[core_var_select_list].astype('category')
    # df_base_core[core_var_select_int_list] = df_base_core[core_var_select_list].apply(lambda x: x.cat.codes)

    # Get Unique Subject/HospAdmission Combinations
    df_ids = df_base_core[['subject_id', 'hadm_id']].drop_duplicates()

    # Get Unique Subject/HospAdmission/Stay Combinations in ICU
    df_icu_ids = df_icustays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    # Get Unique Subject/HospAdmission/Stay Combinations in ED
    df_ed_ids = df_ed_stays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()

    # Save unique subject/hadm_id combinations
    df_ids.to_csv(mimiciv_path + 'mimiciv_patient_adm_ids.csv', index=False)
    df_icu_ids.to_csv(mimiciv_path + 'mimiciv_patient_icu_ids.csv', index=False)
    df_ed_ids.to_csv(mimiciv_path + 'mimiciv_patient_ed_ids.csv', index=False)

    # merge df_ids and df_ed_ids subject_id to get overall length (some patient are only in ed, not admitted)
    all_ids = pd.concat([df_ids[['subject_id']], df_ed_ids[['subject_id']]]).drop_duplicates()

    print(f'Unique Subjects: {len(all_ids)}')
    print('Unique Subjects/HospAdmissions Combinations: ' + str(len(df_ids)))

    # mean and std for number of admissions per patient
    print('Mean number of admissions per patient: ' + str(df_ids.groupby('subject_id').size().mean()))
    print('Std number of admissions per patient: ' + str(df_ids.groupby('subject_id').size().std()))
    # min and max for number of admissions per patient
    print('Min number of admissions per patient: ' + str(df_ids.groupby('subject_id').size().min()))
    print('Max number of admissions per patient: ' + str(df_ids.groupby('subject_id').size().max()))
    # how many patients have only one admission
    print('Number of patients with only one admission: ' + str(sum(df_ids.groupby('subject_id').size() == 1)))


    # GENERATE ALL SINGLE PATIENT RECORDS FOR ENTIRE MIMIC-IV DATABASE
    start = time.time()
    stay_to_ids_dict = generate_all_mimiciv_patient_objects(df_ids, df_icu_ids, df_ed_ids, mimiciv_path, dfs, df_base_core)
    print("Time to generate all patient records: ", time.time() - start)
    print('Overall number of complete stays: ' + str(len(stay_to_ids_dict)))
    print('Stays with only ED stay: ' + str(sum([1 for stay in stay_to_ids_dict.values() if stay['adm_id'] is None])))


if __name__ == '__main__':
    main()
