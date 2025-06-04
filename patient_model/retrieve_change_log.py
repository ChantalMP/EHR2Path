import json
import logging
from copy import deepcopy
import yaml
import pandas as pd

from mimic_iv_extraction.paths import mimiciv_path
from patient_model.retrieve_patient_model import retrieve_patient_model_, get_patient_changelog, restrict_patient_until_hour

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU

def restrict_patient_to_hour(patient_visit, is_ed, hour_idx):
    """
    Restrict the patient_visit to the first hour_idx hours.
    :param patient_visit: the patient_visit dictionary
    :param is_ed: boolean indicating if the patient is in ED at hour_idx
    :param hour_idx: the hour index to which the patient_visit is to be restricted
    :return: the patient_visit dictionary restricted to the first hour_idx hours
    """
    patient_visit_curr = deepcopy(patient_visit)
    if is_ed:
        hour_time = patient_visit_curr.patient_ed.get_data_at_hour(hour_idx)
        patient_visit_curr.patient_adm = None
        patient_visit_curr.patient_icu = None
    else:
        hour_time = patient_visit_curr.patient_adm.get_data_at_hour(hour_idx)
        if patient_visit_curr.patient_icu is not None:
            for idx, _ in enumerate(patient_visit_curr.patient_icu):
                hour_time_icu = patient_visit_curr.patient_icu[idx].get_data_at_hour(hour_idx)
                assert hour_time == hour_time_icu
        patient_visit_curr.patient_ed = None

    return patient_visit_curr, hour_time

def retrieve_change_log_for_hour(patient_visit, current_hour_idx, complete_patient_stay, stay_id=None, add_los=False, los_dict=None, los_only=False):
    """
    Retrieve the change log for the next hour of the patient_visit.
    :param patient_visit: the patient_visit dictionary
    :param current_hour_idx: the current hour index for which the change log is to be retrieved
    :return: the change log describing what changed in the next hour
    """
    next_hour_idx = current_hour_idx + 1
    ed_stay_hours = complete_patient_stay['ed_stay_hours'] if 'ed_stay_hours' in complete_patient_stay else 0
    if ed_stay_hours is None:
        ed_stay_hours = 0
    hosp_stay_hours = complete_patient_stay['hosp_stay_hours'] if 'hosp_stay_hours' in complete_patient_stay else 0
    if hosp_stay_hours is None:
        hosp_stay_hours = 0

    if ed_stay_hours == 0 and hosp_stay_hours == 0:
        return None, None, False, -1

    # Case 1: are in ED and will be next hour -> normal get changelog call
    # Case 2: are in last hour of ED -> discharged or admitted or died?
    # Case 3: within hospital stay -> normal get changelog call
    # Case 4: last hour of hospital stay -> discharged or died?
    if 'ed_stay_hours' in complete_patient_stay and ed_stay_hours != 0 and current_hour_idx == ed_stay_hours: #last hour in ED
        disch_location = patient_visit.patient_ed.ed_stays['disposition'].iloc[0]
        if disch_location == "EXPIRED":
            patient_yaml = {'Emergency Department Stay': {'ICD categories': patient_visit.patient_ed.icd_categories, 'Disposition': 'DIED'}}
            return patient_yaml, None, False, -1
        elif disch_location == "ADMITTED" or (hosp_stay_hours > 0 and patient_visit.patient_adm.dischtime > patient_visit.patient_ed.outtime): # some patients are admitted but have disposition "OTHER" -> check if there is an actual admission
            patient_yaml = {'Emergency Department Stay': {'ICD categories': patient_visit.patient_ed.icd_categories, 'Disposition': 'ADMITTED'}}
            return patient_yaml, None, False, -1
        else:
            patient_yaml = {'Emergency Department Stay': {'ICD categories': patient_visit.patient_ed.icd_categories, 'Disposition': 'DISCHARGED'}}
            return patient_yaml, None, False, -1


    next_hour_in_ed_stay = next_hour_idx <= ed_stay_hours and ed_stay_hours > 0

    if not next_hour_in_ed_stay and ed_stay_hours > 0:
        next_hour_idx = next_hour_idx - ed_stay_hours -2 # want to start again from hour 0 in hospital stay, so need to subtract the ed stay hours, -2 to start from 0 (ended 1 hour after ed end to predict discharge)

    # test if there is a next hour or patient visit is over -> if not, return None
    if not next_hour_in_ed_stay and next_hour_idx > hosp_stay_hours: #last hour in hospital stay
        if hosp_stay_hours != 0:
            disch_location = patient_visit.patient_adm.admissions['discharge_location'].iloc[0] # could do this more fine-grained and include discharged where to
            if disch_location == "DIED":
                patient_yaml = {'Hospital Stay': {'ICD categories': patient_visit.patient_adm.icd_categories, 'Disposition': 'DIED'}}
                return patient_yaml, None, False, -1
            else:
                patient_yaml = {'Hospital Stay': {'ICD categories': patient_visit.patient_adm.icd_categories, 'Disposition': 'DISCHARGED'}}
                return patient_yaml, None, False, -1
        else:
            return None, None, False, -1 # patient neither spend hours in ED nor Hospital

    next_patient_state, next_hour_time = restrict_patient_to_hour(patient_visit, next_hour_in_ed_stay, next_hour_idx) # this is already the change log as patient representation, now need to convert this to text

    entered_icu = False
    released_from_icu_idx = -1
    if patient_visit.patient_icu is not None and len(patient_visit.patient_icu) > 0:
        for idx, stay in enumerate(patient_visit.patient_icu):
            if stay.icustays.iloc[0]['intime'] == next_hour_time:
                entered_icu = True
            elif stay.icustays.iloc[0]['outtime'] == next_hour_time:
                released_from_icu_idx = idx
            else:
                continue

    change_log = get_patient_changelog(patient_visit, next_hour_time, next_patient_state, entered_icu, released_from_icu_idx, add_los=add_los, los_dict=los_dict)

    return change_log, next_patient_state, entered_icu, released_from_icu_idx


'''
Method for integrating the change log into the patient state
Needed for iterative generation of patient states
'''
def integrate_changes_from_changelog(change_log, prior_patient_state, prior_time, full_patient_visit, released_from_icu, entered_icu):
    # full_patient_visit is only used to retrieve units
    new_time = prior_time + pd.Timedelta(1, 'h')
    try: #if structure is not as expected, can't be converted to yaml -> model learns structure quite well, so this should be rare
        yaml_change_log = yaml.safe_load(change_log.strip())
    except yaml.YAMLError:
        print("Could not convert change log to yaml")
        return prior_patient_state
    if 'Emergency Department Stay' in yaml_change_log:
        # ed_vital, ed_pyxis and diagnosis
        yaml_dict = yaml_change_log['Emergency Department Stay']
        # Convert 'Vital Measurements' section back to a DataFrame
        if 'Vital Measurements' in yaml_dict:
            vitals = yaml_dict['Vital Measurements']
            vitals['charttime'] = new_time
            # add row to ed_vital, with the vitals as columns - all not present vitals cols are NaN
            ed_vital = pd.DataFrame([vitals])
            prior_patient_state.patient_ed.ed_vital = pd.concat([prior_patient_state.patient_ed.ed_vital, ed_vital], sort=False, ignore_index=True)
            prior_patient_state.patient_ed.ed_vital = prior_patient_state.patient_ed.ed_vital[['charttime'] + [col for col in prior_patient_state.patient_ed.ed_vital.columns if col != 'charttime']]

        if 'Medication' in yaml_dict:
            medications = yaml_dict['Medication']
            meds = {med: [1] for med in medications if type(med) == str} #filter out wrongly formatted predictions
            ed_pyxis = pd.DataFrame(meds)  # Assuming 1 to indicate medication given
            ed_pyxis.insert(0, 'charttime', new_time)
            prior_patient_state.patient_ed.ed_pyxis = pd.concat([prior_patient_state.patient_ed.ed_pyxis, ed_pyxis], sort=False, ignore_index=True)

        if 'Disposition' in yaml_dict:
            disposition = yaml_dict['Disposition']
            prior_patient_state.patient_ed.ed_stays.disposition = disposition
            prior_patient_state.patient_ed.outtime = new_time - pd.Timedelta(1, 'h')  # disposition is predicted in last hour of ED -> new_time is already in hospital stay
            prior_patient_state.patient_ed.ed_stays.outtime = new_time - pd.Timedelta(1, 'h')
            prior_patient_state.patient_ed.stayhours = (new_time - prior_patient_state.patient_ed.intime).total_seconds() / 3600

        if 'ICD categories' in yaml_dict:
            icd = yaml_dict['ICD categories']
            prior_patient_state.patient_ed.icd_categories = icd

    if 'Hospital Stay' in yaml_change_log:
        if prior_patient_state.patient_adm is None:
            prior_patient_state.patient_adm = Patient_ADM()
            # add empty df
            prior_patient_state.patient_adm.admissions = pd.DataFrame({'admittime': new_time, 'dischtime': None, 'deathtime': None, 'discharge_location': None}, index=[0])
            prior_patient_state.patient_adm.admittime = new_time

        # transfers, services, omr, labevents, microbiologyevents, prescriptions, procedures_icd, disch_notes/diagnosis
        yaml_dict = yaml_change_log['Hospital Stay']
        if 'Patient Location' in yaml_dict:
            transfers = yaml_dict['Patient Location']
            transfers = {t: [1] for t in [transfers]}
            transfers_df = pd.DataFrame(transfers)
            transfers_df.insert(0, 'time', new_time)
            prior_patient_state.patient_adm.transfers = pd.concat([prior_patient_state.patient_adm.transfers, transfers_df], sort=False, ignore_index=True)

        if 'Care Taker' in yaml_dict:
            services = yaml_dict['Care Taker']
            services = {s: [1] for s in [services]}
            services_df = pd.DataFrame(services)
            services_df.insert(0, 'time', new_time)
            prior_patient_state.patient_adm.services = pd.concat([prior_patient_state.patient_adm.services, services_df], sort=False, ignore_index=True)

        if 'Outpatient Measurements' in yaml_dict:
            omr = yaml_dict['Outpatient Measurements']
            omr_time = new_time.replace(hour=0, minute=0, second=0)
            omr_df = pd.DataFrame([omr])
            omr_df.insert(0, 'chartdate', omr_time)
            prior_patient_state.patient_adm.omr = pd.concat([prior_patient_state.patient_adm.omr, omr_df], sort=False, ignore_index=True)

        if 'Lab Results' in yaml_dict:
            labevents = yaml_dict['Lab Results']
            lab_cols = list(labevents.keys())
            if full_patient_visit.patient_adm.labevents is not None:
                for key in lab_cols:
                    if f"{key}_valueuom" in full_patient_visit.patient_adm.labevents.columns: #add units if available
                        all_units = full_patient_visit.patient_adm.labevents[f"{key}_valueuom"].dropna().values
                        if len(all_units) > 0:
                            labevents[f"{key}_valueuom"] = all_units[0]
            labevents_df = pd.DataFrame([labevents])
            labevents_df.insert(0, 'charttime', new_time)
            prior_patient_state.patient_adm.labevents = pd.concat([prior_patient_state.patient_adm.labevents, labevents_df], sort=False, ignore_index=True)

        if 'Microbiology Growth Results' in yaml_dict:
            microbiologyevents = yaml_dict['Microbiology Growth Results']
            elems = []
            for key, value in microbiologyevents.items():
                try:
                    test, spec = key.rsplit(" - ", 1)
                    elems.append({'charttime': new_time, 'test_name': test, 'spec_type_desc': spec, 'org_name': value})
                except Exception as e:
                    print(f"Could not add microbiology event: {key} - {value}")
                    print(e)
                    continue
            microbiologyevents_df = pd.DataFrame(elems)
            prior_patient_state.patient_adm.microbiologyevents = pd.concat([prior_patient_state.patient_adm.microbiologyevents, microbiologyevents_df], sort=False, ignore_index=True)

        if 'Prescriptions' in yaml_dict:
            prescriptions = yaml_dict['Prescriptions']
            if prescriptions is not None: #empty prediction
                prescriptions = {p: [1] for p in prescriptions if type(p) == str} # account for wrongly formatted predictions
                prescriptions_df = pd.DataFrame(prescriptions)
                prescriptions_df.insert(0, 'time', new_time)
                prior_patient_state.patient_adm.prescriptions = pd.concat([prior_patient_state.patient_adm.prescriptions, prescriptions_df], sort=False, ignore_index=True)
                prior_patient_state.patient_adm.prescriptions.fillna(0, inplace=True)

        if 'Procedures' in yaml_dict:
            procedures_icd = yaml_dict['Procedures']
            proc_time = new_time.replace(hour=0, minute=0, second=0)
            procedures_icd_df = pd.DataFrame({'chartdate': proc_time, 'long_title': procedures_icd})
            prior_patient_state.patient_adm.procedures_icd = pd.concat([prior_patient_state.patient_adm.procedures_icd, procedures_icd_df], sort=False, ignore_index=True)

        if 'Disposition' in yaml_dict:
            disposition = yaml_dict['Disposition']
            if disposition != 'ADMITTED TO ICU':
                prior_patient_state.patient_adm.admissions.discharge_location = disposition
                prior_patient_state.patient_adm.admissions.dischtime = new_time if disposition != "DIED" else None
                prior_patient_state.patient_adm.dischtime = new_time if disposition != "DIED" else None
                prior_patient_state.patient_adm.admissions.deathtime = new_time if disposition == "DIED" else None
                prior_patient_state.patient_adm.stayhours = (new_time - prior_patient_state.patient_adm.admittime).total_seconds() / 3600

        if 'ICD categories' in yaml_dict:
            icd = yaml_dict['ICD categories']
            prior_patient_state.patient_adm.icd_categories = icd

        # diagnosis is never in prior patient information, so does not need to be integrated here

    if 'ICU Stay' in yaml_change_log:
        if prior_patient_state.patient_adm is None: #Patient needs to be admitted to be in ICU! -> if not, create admission -> probably very rare that model predicts ICU stay without admission
            if full_patient_visit.patient_adm is None:
                # only happens if model predicts ICU stay for ED only run which is currently not supported / not needed for any downstream task -> return None
                return prior_patient_state
            prior_patient_state.patient_adm = Patient_ADM()
            prior_patient_state.patient_adm.admissions = full_patient_visit.patient_adm.admissions
            prior_patient_state.patient_adm.patients = full_patient_visit.patient_adm.patients
            prior_patient_state.patient_adm.admittime = new_time
            prior_patient_state.patient_adm.admissions.admittime = new_time
            prior_patient_state.patient_adm.admissions.dischtime = None
            prior_patient_state.patient_adm.admissions.deathtime = None
            prior_patient_state.patient_adm.admissions.discharge_location = None

        if prior_patient_state.patient_icu is None:
            prior_patient_state.patient_icu = []

        # inputevents, outputevents, procedureevents, chartevents
        yaml_dict = yaml_change_log['ICU Stay']
        for _, stay in enumerate(yaml_dict):
            stay_idx = stay[-1]
            # if it can be converted to int use it, otherwise just append to the most recent stay
            try:
                stay_idx = int(stay_idx)
            except Exception as e:
                stay_idx = -1
            # if stay_idx is more than current stay + 1, can not add it to the list -> set it to current stay + 1
            if stay_idx > len(prior_patient_state.patient_icu):
                stay_idx = len(prior_patient_state.patient_icu)
            if len(prior_patient_state.patient_icu) == stay_idx: #new entry as ICU object does not exist
                prior_patient_state.patient_icu.append(Patient_ICU())
                prior_patient_state.patient_icu[stay_idx].adm_time = prior_patient_state.patient_adm.admittime
                # create new df with 'intime' 'outtime' and 'los' columns
                prior_patient_state.patient_icu[stay_idx].icustays = pd.DataFrame({'intime': new_time, 'outtime': None, 'los': 0.}, index=[0])

            elif entered_icu: #new enter but ICU object already exists but empty #TODO when do we land here? Test with GT data
                prior_patient_state.patient_icu[stay_idx].adm_time = prior_patient_state.patient_adm.admittime
                prior_patient_state.patient_icu[stay_idx].icustays['intime'] = new_time

            orig_patient_was_in_icu = full_patient_visit.patient_icu is not None and stay_idx < len(full_patient_visit.patient_icu)
            try:
                stay_dict = yaml_dict[stay]
            except Exception as e:
                logging.warning(f"Could not convert stay to index: {stay}")
                stay_dict = None

            if stay_dict is not None: # emtpy stay prediction
                if 'Medication' in stay_dict:
                    medications = stay_dict['Medication']
                    meds = {med: [1] for med in medications if type(med) == str}
                    inputevents = pd.DataFrame(meds, index=[0])
                    inputevents.insert(0, 'time', new_time)
                    prior_patient_state.patient_icu[stay_idx].inputevents = pd.concat([prior_patient_state.patient_icu[stay_idx].inputevents, inputevents], sort=False, ignore_index=True)
                    prior_patient_state.patient_icu[stay_idx].inputevents.fillna(0, inplace=True)

                if 'Output' in stay_dict:
                    outputevents = stay_dict['Output']
                    out_cols = list(outputevents.keys()) if type(outputevents) == dict else []
                    if orig_patient_was_in_icu and full_patient_visit.patient_icu[stay_idx].outputevents is not None: # add units if available
                        for key in out_cols:
                            if f"{key}_valueuom" in full_patient_visit.patient_icu[stay_idx].outputevents.columns:
                                all_units = full_patient_visit.patient_icu[stay_idx].outputevents[f"{key}_valueuom"].dropna().values
                                if len(all_units) > 0:
                                    outputevents[f"{key}_valueuom"] = all_units[0]

                    outputevents_df = pd.DataFrame([outputevents]) if type(outputevents) == dict else pd.DataFrame()
                    outputevents_df.insert(0, 'charttime', new_time)
                    prior_patient_state.patient_icu[stay_idx].outputevents = pd.concat([prior_patient_state.patient_icu[stay_idx].outputevents, outputevents_df], sort=False, ignore_index=True)

                if 'Procedures' in stay_dict:
                    procedures = stay_dict['Procedures']
                    procedures = {p: [1] for p in procedures if type(procedures) == str}
                    procedures_df = pd.DataFrame(procedures)
                    procedures_df.insert(0, 'time', new_time)
                    prior_patient_state.patient_icu[stay_idx].procedureevents = pd.concat([prior_patient_state.patient_icu[stay_idx].procedureevents, procedures_df], sort=False, ignore_index=True)
                    prior_patient_state.patient_icu[stay_idx].procedureevents.fillna(0, inplace=True)

                if 'Chart Events' in stay_dict and type(stay_dict['Chart Events']) == dict:
                    for event in stay_dict['Chart Events']:
                        chartevents = stay_dict['Chart Events'][event]
                        chart_cols = list(chartevents.keys()) if type(chartevents) == dict else []
                        if orig_patient_was_in_icu and full_patient_visit.patient_icu[stay_idx].chartevents is not None and event in full_patient_visit.patient_icu[stay_idx].chartevents: # add units if available
                            for key in chart_cols:
                                if full_patient_visit.patient_icu[stay_idx].chartevents[event] is not None and f"{key}_valueuom" in full_patient_visit.patient_icu[stay_idx].chartevents[event].columns:
                                    all_units = full_patient_visit.patient_icu[stay_idx].chartevents[event][f"{key}_valueuom"].dropna().values
                                    if len(all_units) > 0:
                                        chartevents[f"{key}_valueuom"] = all_units[0]
                        chartevents_df = pd.DataFrame([chartevents]) if type(chartevents) == dict else pd.DataFrame()
                        chartevents_df.insert(0, 'charttime', new_time)
                        if prior_patient_state.patient_icu[stay_idx].chartevents is None:
                            prior_patient_state.patient_icu[stay_idx].chartevents = {}
                        if event not in prior_patient_state.patient_icu[stay_idx].chartevents:
                            prior_patient_state.patient_icu[stay_idx].chartevents[event] = None
                        prior_patient_state.patient_icu[stay_idx].chartevents[event] = pd.concat([prior_patient_state.patient_icu[stay_idx].chartevents[event], chartevents_df], sort=False, ignore_index=True)

    if released_from_icu and 'ICU Stay' in yaml_change_log: #if patient never was in ICU, but is discharged from hospital, this flag is also positive
        # add outtime and los to last ICU stay
        if prior_patient_state.patient_icu is not None and len(prior_patient_state.patient_icu) > 0:
            prior_patient_state.patient_icu[-1].icustays['outtime'] = new_time
            los_in_days = prior_patient_state.patient_icu[-1].icustays['outtime'] - prior_patient_state.patient_icu[-1].icustays['intime']
            # convert to fractional days
            prior_patient_state.patient_icu[-1].icustays['los'] = los_in_days.iloc[0].total_seconds() / 86400

    return prior_patient_state

'''
very similar to above, but returns the extracted changes as dataframes instead of integrating them into the prior patient state
'''
def extract_changes_from_changelog(change_log, full_patient_visit, eval_future_ed_disp=False, eval_future_mort=False, eval_future_los=False):
    # full_patient_visit is only used to retrieve units
    try: #if structure is not as expected, can't be converted to yaml
        yaml_change_log = yaml.safe_load(change_log)
        if yaml_change_log is None:
            print("yaml_change_log is None: ")
            print(change_log)
    except Exception as e:
        yaml_change_log = None
        print(e)

    if yaml_change_log is None or type(yaml_change_log) != dict:
        print("Could not convert change log to yaml")
        ed_vital = pd.DataFrame()
        ed_pyxis = pd.DataFrame()
        ed_icd = None
        ed_disposition = None
        ed_los = None

        transfers_df = pd.DataFrame()
        services_df = pd.DataFrame()
        labevents_df = pd.DataFrame()
        microbiologyevents_df = pd.DataFrame()
        prescriptions_df = pd.DataFrame()
        procedures_icd_df = pd.DataFrame()
        icd = None
        disposition = None
        hospital_los = None

        inputevents = pd.DataFrame()
        outputevents_df = pd.DataFrame()
        procedures_df = pd.DataFrame()
        chartevents_dict = {}
        disposition_icu = None
        icu_los = None

        return ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df, procedures_icd_df, icd, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict, disposition_icu, ed_los, hospital_los, icu_los


    if 'Emergency Department Stay' in yaml_change_log and yaml_change_log['Emergency Department Stay'] is not None:
        # ed_vital, ed_pyxis and diagnosis
        yaml_dict = yaml_change_log['Emergency Department Stay']

        ed_los = yaml_dict['LOS'] if 'LOS' in yaml_dict else None

        # Convert 'Vital Measurements' section back to a DataFrame
        try:
            if 'Vital Measurements' in yaml_dict:
                vitals = yaml_dict['Vital Measurements']
                # add row to ed_vital, with the vitals as columns - all not present vitals cols are NaN
                ed_vital = pd.DataFrame([vitals])
            else:
                ed_vital = pd.DataFrame() # no changes in vitals predicted
        except Exception as e:
            print("Could not convert vitals to dataframe")
            print(e)
            ed_vital = pd.DataFrame()


        if 'Medication' in yaml_dict:
            try:
                medications = yaml_dict['Medication']
                meds = {med: [1] for med in medications}
                ed_pyxis = pd.DataFrame(meds)  # Assuming 1 to indicate medication given
            except Exception as e:
                print("Could not convert medications to dataframe")
                print(e)
                ed_pyxis = pd.DataFrame()
        else:
            ed_pyxis = pd.DataFrame()
        if 'ICD categories' in yaml_dict:
            try:
                ed_icd = yaml_dict['ICD categories'].split(";")
            except Exception as e:
                print("Could not convert ICD to dataframe")
                print(e)
                ed_icd = None
        else:
            ed_icd = None

        if 'Disposition' in yaml_dict:
            ed_disposition = yaml_dict['Disposition']
        elif eval_future_ed_disp and 'Future Disposition' in yaml_dict:
            ed_disposition = yaml_dict['Future Disposition']
        else:
            ed_disposition = None

    else: # no ED changes
        ed_vital = pd.DataFrame()
        ed_pyxis = pd.DataFrame()
        ed_icd = None
        ed_disposition = None
        ed_los = None

    if 'Hospital Stay' in yaml_change_log and yaml_change_log['Hospital Stay'] is None:
        print("Hospital Stay is None")
        print(yaml_change_log)
    if 'Hospital Stay' in yaml_change_log and yaml_change_log['Hospital Stay'] is not None:
        # transfers, services, omr, labevents, microbiologyevents, prescriptions, procedures_icd, disch_notes/diagnosis
        yaml_dict = yaml_change_log['Hospital Stay']

        hospital_los = yaml_dict['LOS'] if 'LOS' in yaml_dict else None

        try:
            if 'Patient Location' in yaml_dict:
                transfers = yaml_dict['Patient Location']
                transfers = {t: [1] for t in [transfers]}
                transfers_df = pd.DataFrame(transfers)
            else:
                transfers_df = pd.DataFrame()
        except Exception as e:
            print("Could not convert transfers to dataframe")
            print(e)
            transfers_df = pd.DataFrame()

        if 'Care Taker' in yaml_dict:
            try:
                services = yaml_dict['Care Taker']
                services = {s: [1] for s in [services]}
                services_df = pd.DataFrame(services)
            except Exception as e:
                print("Could not convert services to dataframe")
                print(e)
                services_df = pd.DataFrame()
        else:
            services_df = pd.DataFrame()

        if 'Lab Results' in yaml_dict:
            labevents = yaml_dict['Lab Results']
            try:
                lab_cols = list(labevents.keys())
                if full_patient_visit.patient_adm is not None and full_patient_visit.patient_adm.labevents is not None: # does GT patient have this df -> only then can we extract the unit
                    for key in lab_cols:
                        if f"{key}_valueuom" in full_patient_visit.patient_adm.labevents.columns:
                            all_units = full_patient_visit.patient_adm.labevents[f"{key}_valueuom"].dropna().values
                            if len(all_units) > 0:
                                labevents[f"{key}_valueuom"] = all_units[0]
                labevents_df = pd.DataFrame([labevents])
            except Exception as e:
                print("Could not convert labevents to dataframe")
                print(e)
                labevents_df = pd.DataFrame()
        else:
            labevents_df = pd.DataFrame()

        if 'Microbiology Growth Results' in yaml_dict:
            try:
                microbiologyevents = yaml_dict['Microbiology Growth Results']
                elems = []
                for key, value in microbiologyevents.items():
                    if " - " in key:
                        test, spec = key.rsplit(" - ", 1)
                    else: # for predicted outputs this might happen
                        test = key
                        spec = "unknown"
                    # if " antibiotics: " in value:
                    #     growth, ab = value.split(" antibiotics: ")
                    #     elems.append({'test_name': test, 'spec_type_desc': spec, 'org_name': growth, 'ab_interpretation': ab})
                    # else:
                    #     growth = value
                    elems.append({'test_name': test, 'spec_type_desc': spec, 'org_name': value})
                microbiologyevents_df = pd.DataFrame(elems)
            except Exception as e:
                print("Could not convert microbiologyevents to dataframe")
                print(e)
                microbiologyevents_df = pd.DataFrame()
        else:
            microbiologyevents_df = pd.DataFrame()

        if 'Prescriptions' in yaml_dict:
            try:
                prescriptions = yaml_dict['Prescriptions']
                prescriptions = {p: [1] for p in prescriptions}
                prescriptions_df = pd.DataFrame(prescriptions)
            except Exception as e:
                print("Could not convert prescriptions to dataframe")
                print(e)
                prescriptions_df = pd.DataFrame()
        else:
            prescriptions_df = pd.DataFrame()

        if 'Procedures' in yaml_dict:
            try:
                procedures_icd = yaml_dict['Procedures']
                procedures_icd_df = pd.DataFrame({'long_title': procedures_icd})
            except Exception as e:
                print("Could not convert procedures to dataframe")
                print(e)
                procedures_icd_df = pd.DataFrame()
        else:
            procedures_icd_df = pd.DataFrame()

        if 'ICD categories' in yaml_dict:
            try:
                icd = yaml_dict['ICD categories'].split(";")
            except Exception as e:
                print("Could not convert ICD to dataframe")
                print(e)
                icd = None
        else:
            icd = None

        if 'Disposition' in yaml_dict:
            disposition = yaml_dict['Disposition']
        else:
            disposition = None

    else: # no hospital changes
        transfers_df = pd.DataFrame()
        services_df = pd.DataFrame()
        labevents_df = pd.DataFrame()
        microbiologyevents_df = pd.DataFrame()
        prescriptions_df = pd.DataFrame()
        procedures_icd_df = pd.DataFrame()
        icd = None
        disposition = None
        hospital_los = None

    if 'ICU Stay' in yaml_change_log and yaml_change_log['ICU Stay'] is not None:
        # inputevents, outputevents, procedureevents, chartevents
        yaml_dict = yaml_change_log['ICU Stay']
        inputevents_all = {}
        outputevents_df_all = {}
        procedures_df_all = {}
        chartevents_dict_all = {}
        disposition_icu_all = {}
        icu_los_all = {}

        for idx, stay in enumerate(yaml_dict):
            try:
                stay_idx = int(stay[-1])
                stay_dict = yaml_dict[stay]
            except Exception as e:
                logging.warning(f"Could not convert stay to index: {stay}")
                continue
            if stay_dict is None:
                continue

            icu_los = stay_dict['LOS'] if 'LOS' in stay_dict else None
            icu_los_all[stay] = icu_los

            if 'Medication' in stay_dict:
                try:
                    medications = stay_dict['Medication']
                    meds = {med: [1] for med in medications}
                    inputevents = pd.DataFrame(meds, index=[0])
                except Exception as e:
                    print("Could not convert inputevents to dataframe")
                    print(e)
                    inputevents = pd.DataFrame()
            else:
                inputevents = pd.DataFrame()

            inputevents_all[stay] = inputevents

            if 'Output' in stay_dict:
                try:
                    outputevents = stay_dict['Output']
                    out_cols = list(outputevents.keys()) if type(outputevents) == dict else []
                    if full_patient_visit.patient_icu is not None and stay_idx < len(full_patient_visit.patient_icu) and full_patient_visit.patient_icu[stay_idx].outputevents is not None: # does GT patient have this df -> only then can we extract the unit
                        for key in out_cols:
                            if f"{key}_valueuom" in full_patient_visit.patient_icu[stay_idx].outputevents.columns:
                                all_units = full_patient_visit.patient_icu[stay_idx].outputevents[f"{key}_valueuom"].dropna().values
                                if len(all_units) > 0:
                                    outputevents[f"{key}_valueuom"] = full_patient_visit.patient_icu[stay_idx].outputevents[f"{key}_valueuom"].dropna().values[0]

                    outputevents_df = pd.DataFrame([outputevents]) if type(outputevents) == dict else pd.DataFrame()
                except Exception as e:
                    print("Could not convert outputevents to dataframe")
                    print(e)
                    outputevents_df = pd.DataFrame()
            else:
                outputevents_df = pd.DataFrame()

            outputevents_df_all[stay] = outputevents_df

            if 'Procedures' in stay_dict:
                try:
                    procedures = stay_dict['Procedures']
                    procedures = {p: [1] for p in procedures}
                    procedures_df = pd.DataFrame(procedures)
                except Exception as e:
                    print("Could not convert procedures to dataframe")
                    print(e)
                    procedures_df = pd.DataFrame
            else:
                procedures_df = pd.DataFrame()

            procedures_df_all[stay] = procedures_df

            chartevents_dict = {}
            if 'Chart Events' in stay_dict  and stay_dict['Chart Events'] is not None:
                for event in stay_dict['Chart Events']:
                    try:
                        chartevents = stay_dict['Chart Events'][event]
                        chart_cols = list(chartevents.keys()) if type(chartevents) == dict else []
                        if full_patient_visit.patient_icu is not None and stay_idx in full_patient_visit.patient_icu and full_patient_visit.patient_icu[
                            stay_idx].chartevents is not None and event in full_patient_visit.patient_icu[stay_idx].chartevents:  # does GT patient have this df -> only then can we extract the unit
                            for key in chart_cols:
                                if f"{key}_valueuom" in full_patient_visit.patient_icu[stay_idx].chartevents[event].columns:
                                    all_units = full_patient_visit.patient_icu[stay_idx].chartevents[event][f"{key}_valueuom"].dropna().values
                                    if len(all_units) > 0:
                                        chartevents[f"{key}_valueuom"] = full_patient_visit.patient_icu[stay_idx].chartevents[event][f"{key}_valueuom"].dropna().values[0]
                        chartevents_df = pd.DataFrame([chartevents]) if type(chartevents) == dict else pd.DataFrame()
                        chartevents_dict[event] = chartevents_df
                    except Exception as e:
                        print(f"Could not convert chartevents {event} to dataframe")
                        print(e)
                        if type(event) == str:
                            chartevents_dict[event] = pd.DataFrame()

            chartevents_dict_all[stay] = chartevents_dict

            if 'Disposition' in stay_dict:
                disposition_icu = stay_dict['Disposition']
            elif eval_future_mort and '24h Disposition' in stay_dict:
                disposition = stay_dict['24h Disposition'] #set final dispostition -> if DIED will be labeled/predicted as 1, all else 0 (so "ALIVE" is fine)
                disposition_icu = None
            elif eval_future_los and 'StayOver3days' in stay_dict:
                disposition = stay_dict['StayOver3days'] # "DISCHARGED" or "STAYED"
                disposition = "STAYED" if disposition == 'YES' else 'DISCHARGED'
                disposition_icu = None
            elif eval_future_los and '3day LOS' in stay_dict:
                disposition = stay_dict['3day LOS'] # "DISCHARGED" or "STAYED"
                disposition_icu = None
            else:
                disposition_icu = None

            disposition_icu_all[stay] = disposition_icu

    else: # no ICU changes
        inputevents_all = pd.DataFrame()
        outputevents_df_all = pd.DataFrame()
        procedures_df_all = pd.DataFrame()
        chartevents_dict_all = {}
        disposition_icu_all = None
        icu_los_all = None

    return ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df, procedures_icd_df, icd, disposition, inputevents_all, outputevents_df_all, procedures_df_all, chartevents_dict_all, disposition_icu_all, ed_los, hospital_los, icu_los_all



if __name__ == '__main__':
    pass