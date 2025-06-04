import random

import pandas as pd


def create_mean_majority_changelog(mean_maj_dict):
    # ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df, procedures_icd_df, icd, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict
    ed_vital = pd.DataFrame()

    labevents_df = pd.DataFrame()

    outputevents_df = pd.DataFrame()
    procedures_df = pd.DataFrame()
    chartevents_dict = {}

    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('vital'):
            col = key.split('_')[-1]
            val = value['mean'] if 'mean' in value else value['majority']
            ed_vital[col] = [val]

    # ed pyxis
    pyxis_total = round(mean_maj_dict['pyxis_total']['mean'])
    meds = mean_maj_dict['pyxis']['majority'][:pyxis_total]
    ed_pyxis = pd.DataFrame({med: [1] for med in meds})

    # ed disposition
    ed_disposition = mean_maj_dict['ed_disposition']['majority']

    # ed icd
    ed_icd_total = round(mean_maj_dict['ed_icd_categories_total']['mean'])
    ed_icd = mean_maj_dict['ed_icd_categories']['majority'][:ed_icd_total]

    # hospital stay
    # transfers
    transfer = mean_maj_dict['transfers']['majority']
    transfers_df = pd.DataFrame({transfer: [1]})

    # services
    service = mean_maj_dict['services']['majority']
    services_df = pd.DataFrame({service: [1]})

    # labevents
    updates = {}
    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('lab'):
            col = key.split('_')[-1]
            val = value.get('mean', value.get('majority'))  # Get 'mean' if it exists, else 'majority'
            updates[col] = [val]  # Add to updates dictionary

    # Apply all updates at once using pd.DataFrame and pd.concat
    if updates:  # Check if there are any updates to apply
        new_columns = pd.DataFrame(updates)
        labevents_df = pd.concat([labevents_df, new_columns], axis=1)

    # microbiologyevents
    microbiology = mean_maj_dict['microbiology']['majority']
    test, spec, value = microbiology.split(" --- ")
    elems = [{'test_name': test, 'spec_type_desc': spec, 'org_name': value}]
    microbiologyevents_df = pd.DataFrame(elems)

    # prescriptions
    prescriptions_total = round(mean_maj_dict['prescriptions_total']['mean'])
    prescriptions = mean_maj_dict['prescriptions']['majority'][:prescriptions_total]
    prescriptions_df = pd.DataFrame({p: [1] for p in prescriptions})

    # procedures
    procedure = mean_maj_dict['procedures']['majority']
    procedures_icd_df = pd.DataFrame({'long_title': [procedure]})

    # disposition
    disposition = mean_maj_dict['disposition']['majority']

    # icd
    idc_total = round(mean_maj_dict['icd_categories_total']['mean'])
    icd = mean_maj_dict['icd_categories']['majority'][:idc_total]

    # ICU stay
    # inputevents
    inputevents_total = round(mean_maj_dict['inputevents_total']['mean'])
    inputevents = mean_maj_dict['inputevents']['majority'][:inputevents_total]
    inputevents = pd.DataFrame({i: [1] for i in inputevents})

    # outputevents
    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('output'):
            col = key.split('_')[-1]
            val = value['mean'] if 'mean' in value else value['majority']
            outputevents_df[col] = [val]

    # procedures
    procedureevents_total = round(mean_maj_dict['procedureevents_total']['mean'])
    procedureevents = mean_maj_dict['procedureevents']['majority'][:procedureevents_total]
    procedures_df = pd.DataFrame({p: [1] for p in procedureevents})

    # chartevents
    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('charts'):
            event, col = key[7:].split(' --- ')
            val = value['mean'] if 'mean' in value else value['majority']
            if event not in chartevents_dict:
                chartevents_dict[event] = pd.DataFrame()
            chartevents_dict[event][col] = [val]

    return ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df, procedures_icd_df, icd, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict

def create_event_freq_changelog(event_freq, mean_maj_dict):
    # ed_vital, ed_pyxis, ed_icd, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df, procedures_icd_df, icd, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict
    ed_vital = pd.DataFrame()

    labevents_df = pd.DataFrame()

    outputevents_df = pd.DataFrame()
    chartevents_dict = {}

    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('vital_'):
            col = key.split('_')[-1]
            ratio = event_freq[f'Emergency Department Stay.Vital Measurements.{col}']
            rand = random.random()
            if rand < ratio:
                val = value['mean'] if 'mean' in value else value['majority']
                ed_vital[col] = [val]

    # ed pyxis
    meds = []
    for key, value in event_freq.items():
        if key.startswith('Emergency Department Stay.Medication'):
            med = key.split('.')[-1]
            ratio = event_freq[key]
            rand = random.random()
            if rand < ratio:
                meds.append(med)

    ed_pyxis = pd.DataFrame({med: [1] for med in meds})

    # ed disposition
    ratio = event_freq['Emergency Department Stay.Disposition']
    rand = random.random()
    if rand < ratio:
        ed_disposition = mean_maj_dict['ed_disposition']['majority'] #no matter which disposition as only event occurance is relevant for this baseline
    else:
        ed_disposition = None

    # hospital stay
    # transfers
    ratio = event_freq['Hospital Stay.Patient Location']
    rand = random.random()
    if rand < ratio:
        transfer = mean_maj_dict['transfers']['majority']
        transfers_df = pd.DataFrame({transfer: [1]})
    else:
        transfers_df = pd.DataFrame()

    # services
    ratio = event_freq['Hospital Stay.Care Taker']
    rand = random.random()
    if rand < ratio:
        service = mean_maj_dict['services']['majority']
        services_df = pd.DataFrame({service: [1]})
    else:
        services_df = pd.DataFrame()

    # labevents
    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('lab'):
            col = key.split('_')[-1]
            ratio = event_freq[f'Hospital Stay.Lab Results.{key.split("_")[-1]}'] if f'Hospital Stay.Lab Results.{key.split("_")[-1]}' in event_freq else 0
            rand = random.random()
            if rand < ratio:
                val = value['mean'] if 'mean' in value else value['majority']
                labevents_df[col] = [val]

    # microbiologyevents

    tests = []
    for key, value in event_freq.items():
        if key.startswith('Hospital Stay.Microbiology Growth Results'):
            elem = key.split('.')[-1]
            ratio = event_freq[key]
            rand = random.random()
            if rand < ratio:
                tests.append(elem)

    elems = []
    microbiology = mean_maj_dict['microbiology']['majority']
    test, spec, value = microbiology.split(" --- ")
    for test in tests:
        elem = {'test_name': test, 'spec_type_desc': spec, 'org_name': value} #org_name is categorical value, 'no
        elems.append(elem)
    microbiologyevents_df = pd.DataFrame(elems)

    # prescriptions
    prescriptions = []
    for key, value in event_freq.items():
        if key.startswith('Hospital Stay.Prescriptions'):
            med = key.split('.')[-1]
            ratio = event_freq[key]
            rand = random.random()
            if rand < ratio:
                prescriptions.append(med)
    prescriptions_df = pd.DataFrame({p: [1] for p in prescriptions})

    # procedures
    procedures = []
    for key, value in event_freq.items():
        if key.startswith('Hospital Stay.Procedures'):
            proc = key.split('.')[-1]
            ratio = event_freq[key]
            rand = random.random()
            if rand < ratio:
                procedures.append(proc)
    procedures_icd_df = pd.DataFrame({'long_title': procedures})

    # disposition
    ratio = event_freq['Hospital Stay.Disposition']
    rand = random.random()
    if rand < ratio:
        disposition = mean_maj_dict['disposition']['majority']  # no matter which disposition as only event occurrence is relevant for this baseline
    else:
        disposition = None

    # ICU stay
    # inputevents
    inputevents = []
    for key, value in event_freq.items():
        if key.startswith('ICU Stay.Medication'):
            med = key.split('.')[-1]
            ratio = event_freq[key]
            rand = random.random()
            if rand < ratio:
                inputevents.append(med)
    inputevents = pd.DataFrame({i: [1] for i in inputevents})

    # outputevents
    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('output'):
            col = key.split('_')[-1]
            ratio = event_freq[f'ICU Stay.Output.{col}'] if f'ICU Stay.Output.{col}' in event_freq else 0
            rand = random.random()
            if rand < ratio:
                val = value['mean'] if 'mean' in value else value['majority']
                outputevents_df[col] = [val]

    # procedures
    procedureevents = []
    for key, value in event_freq.items():
        if key.startswith('ICU Stay.Procedures'):
            proc = key.split('.')[-1]
            ratio = event_freq[key]
            rand = random.random()
            if rand < ratio:
                procedureevents.append(proc)
    procedures_df = pd.DataFrame({p: [1] for p in procedureevents})

    # chartevents
    for key, value in mean_maj_dict.items():
        # ed vital signs
        if key.startswith('charts'):
            event, col = key[7:].split(' --- ')
            ratio = event_freq[f'ICU Stay.Chart Events.{event}.{col}'] if f'ICU Stay.Chart Events.{event}.{col}' in event_freq else 0
            rand = random.random()
            if rand < ratio:
                val = value['mean'] if 'mean' in value else value['majority']
                if event not in chartevents_dict:
                    chartevents_dict[event] = pd.DataFrame()
                chartevents_dict[event][col] = [val]

    return ed_vital, ed_pyxis, None, ed_disposition, transfers_df, services_df, labevents_df, microbiologyevents_df, prescriptions_df, procedures_icd_df, None, disposition, inputevents, outputevents_df, procedures_df, chartevents_dict