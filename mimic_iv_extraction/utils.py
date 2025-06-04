from copy import deepcopy

from convert_column import convert_column_to_sorted_periods_string_cython, convert_column_to_time_events_string_cython
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import notna

from mimic_iv_extraction.mappings.service_mapping import service_mapping

noon_time = datetime.strptime('12:00', '%H:%M').time()


def convert_ts_to_columns(df, label_col_name='label', value_col_name='value', secondary_value_col=None, third_value_col=None, info_cols=None, time_col_name="", aggfunc='first'):
    # merge value_cols if necessary
    if secondary_value_col is not None:
        df['combined_value'] = df[value_col_name].combine_first(df[secondary_value_col]) #replace if NaN
        value_col_name = 'combined_value'
        if third_value_col is not None: # only if first and secondary where NaN
            df['combined_value2'] = df['combined_value'].combine_first(df[third_value_col])

    if aggfunc == 'sum' or aggfunc == 'mean': #convert to numeric
        df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')
    # Create pivot table for the main label and value columns
    pivot_df = df.pivot_table(index=time_col_name, columns=label_col_name, values=value_col_name, aggfunc=aggfunc)

    if info_cols is not None:
        # Handle additional info columns
        for col in info_cols:
            info_pivot_df = df.pivot_table(index=time_col_name, columns=label_col_name, values=col, aggfunc='first')
            # Rename columns to match the desired format
            info_pivot_df.columns = [f'{label}_{col}' for label in info_pivot_df.columns]
            # Merge info columns with the main pivot table
            pivot_df = pivot_df.join(info_pivot_df)

    # Remove rows where all added columns are NaN
    df_out = pivot_df.dropna(axis=0, how='all').reset_index()

    return df_out


def convert_microbiology(df):
    interpretation_map = {
        "S": "sensitive",
        "R": "resistant",
        "I": "intermediate",
        "P": "pending",
        "none": "none"
    }

    # Convert interpretation letters to words
    df["interpretation"] = df["interpretation"].map(interpretation_map)

    # Grouping and aggregating
    def aggregate_interpretations(group):
        interpretation_dict = defaultdict(list)
        for ab_name, interpretation in zip(group["ab_name"], group["interpretation"]):
            if ab_name.lower() != "none":
                interpretation_dict[interpretation].append(ab_name)
        return ', '.join(f"{interpretation}: {list(ab_names)}" for interpretation, ab_names in interpretation_dict.items()) if interpretation_dict else np.nan

    summary_df = (df.groupby(["charttime", "test_name", "org_name", "spec_type_desc"])
                  .apply(aggregate_interpretations)
                  .reset_index(name="ab_interpretation"))

    return summary_df


def create_hourly_event_df(df, label_col_name='label', start_time_col='start_time', end_time_col='end_time', time_col_name='hour', info_cols=None, disch_time=None, admit_time=None):
    if df is None or df.empty:
        return df
    # Create a date range for the full period with hourly frequency
    min_time = df[start_time_col].min().floor('h')
    if pd.isnull(min_time): # very rarely for prescriptions start or end time is not set for any medication, we then assume min/max start/end time as admit/discharge time
        min_time = admit_time
    max_time = df[end_time_col].max().ceil('h')
    # check if max_time is NaT
    if pd.isnull(max_time):
            max_time = disch_time
    all_hours = pd.date_range(start=min_time, end=max_time, freq='h')

    # Initialize the result dataframe
    result_df = pd.DataFrame({time_col_name: all_hours})

    # For each unique label, create a column with 0/1 indicating activity
    unique_labels = df[label_col_name].unique()
    unique_labels = unique_labels[~pd.isnull(unique_labels)]  # Remove NaN labels

    for label in unique_labels:
        result_df[label] = 0  # Initialize column with 0

        # Add columns for each info column
        if info_cols is not None:
            new_columns = {f'{label}_{col}': np.nan for col in info_cols}
            new_df = pd.DataFrame(new_columns, index=result_df.index)
            result_df = pd.concat([result_df, new_df], axis=1)

        # Find the rows corresponding to the current label
        label_rows = df[df[label_col_name] == label]

        for _, row in label_rows.iterrows():
            # Create a date range for the event duration
            if pd.isna(row[end_time_col]) and pd.isna(row[start_time_col]):
                # both are NaT - assume error / not administered
                continue
            elif pd.isna(row[end_time_col]):
                row[end_time_col] = row[start_time_col] + pd.Timedelta(hours=1)
            elif pd.isna(row[start_time_col]):
                row[start_time_col] = row[end_time_col] - pd.Timedelta(hours=1)
            else:
                pass
            event_hours = pd.date_range(start=row[start_time_col].round('h'), end=row[end_time_col].round('h'), freq='h')

            # Set the corresponding hours to 1
            result_df.loc[result_df[time_col_name].isin(event_hours), label] = 1

            # Set the info columns for the corresponding hours
            if info_cols is not None:
                for col in info_cols:
                    result_df.loc[result_df[time_col_name].isin(event_hours), f'{label}_{col}'] = row[col]

    return result_df


def last(x):
    arr = x.array[notna(x.array)]
    if not len(arr):
        return x.array.dtype.na_value
    return arr[-1]

def mean_or_last_agg(series):
    # Attempt to convert to numeric and compute the mean
    numeric_series = pd.to_numeric(series, errors='coerce')
    # Check if there are any valid numeric values
    if numeric_series.notna().any():
        return numeric_series.mean()
    else:
        return last(series)


# from https://github.com/healthylaife/MIMIC-IV-Data-Pipeline
def ndc_meds(med) -> pd.DataFrame:
    # Convert any nan values to a dummy value
    med.ndc = med.ndc.fillna(-1)

    # Ensures the decimal is removed from the ndc col
    med.ndc = med.ndc.astype("Int64")

    # The NDC codes in the prescription dataset is the 11-digit NDC code, although codes are missing
    # their leading 0's because the column was interpreted as a float then integer; this function restores
    # the leading 0's, then obtains only the PRODUCT and MANUFACTUERER parts of the NDC code (first 9 digits)
    def to_str(ndc):
        if ndc < 0:  # dummy values are < 0
            return np.nan
        ndc = str(ndc)
        return (("0" * (11 - len(ndc))) + ndc)[0:-2]

    # The mapping table is ALSO incorrectly formatted for 11 digit NDC codes. An 11 digit NDC is in the
    # form of xxxxx-xxxx-xx for manufactuerer-product-dosage. The hyphens are in the correct spots, but
    # the number of digits within each section may not be 5-4-2, in which case we add leading 0's to each
    # to restore the 11 digit format. However, we only take the 5-4 sections, just like the to_str function
    def format_ndc_table(ndc):
        parts = ndc.split("-")
        return ("0" * (5 - len(parts[0])) + parts[0]) + ("0" * (4 - len(parts[1])) + parts[1])

    def read_ndc_mapping2(map_path):
        ndc_map = pd.read_csv(map_path, header=0, delimiter='\t', encoding='latin1')
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
        ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
        ndc_map.columns = list(map(str.lower, ndc_map.columns))
        return ndc_map

    # Read in NDC mapping table
    ndc_map = read_ndc_mapping2("mimic_iv_extraction/mappings/ndc_product.txt")[['productndc', 'nonproprietaryname', 'pharm_classes']]

    # Normalize the NDC codes in the mapping table so that they can be merged
    ndc_map['new_ndc'] = ndc_map.productndc.apply(format_ndc_table)
    ndc_map.drop_duplicates(subset=['new_ndc', 'nonproprietaryname'], inplace=True)
    med['new_ndc'] = med.ndc.apply(to_str)

    # Left join the med dataset to the mapping information
    med = med.merge(ndc_map, how='inner', left_on='new_ndc', right_on='new_ndc')

    # In NDC mapping table, the pharm_class col is structured as a text string, separating different pharm classes from eachother
    # This can be [PE], [EPC], and others, but we're interested in EPC. Luckily, between each commas, it states if a phrase is [EPC]
    # So, we just string split by commas and keep phrases containing "[EPC]"
    def get_EPC(s):
        """Gets the Established Pharmacologic Class (EPC) from the mapping table"""
        if type(s) != str:
            return np.nan
        words = s.split(",")
        return [x for x in words if "[EPC]" in x]

    # Function generates a list of EPCs, as a drug can have multiple EPCs
    med['EPC'] = med.pharm_classes.apply(get_EPC)

    return med


def convert_table_to_sorted_periods_string(df, curr_time, use_service_mapping=False):
    df = df.set_index('time')
    periods = []

    for column in df.columns:
        active_periods = df[df[column] == 1].index
        if not active_periods.empty:
            start_time = None
            for i, time_point in enumerate(active_periods):
                if start_time is None:
                    start_time = time_point
                if i < len(active_periods) - 1 and (active_periods[i + 1] - time_point).total_seconds() // 3600 != 1:
                    end_time = time_point
                    periods.append((start_time, end_time, column))
                    start_time = None
                elif i == len(active_periods) - 1:
                    end_time = time_point
                    periods.append((start_time, end_time, column))

    periods.sort(key=lambda x: x[0])
    result = []
    for start, end, department in periods:
        start_time = f"{int((curr_time - start)/ pd.Timedelta(hours=1))}"
        end_time = f"{int((curr_time - end)/ pd.Timedelta(hours=1))}"
        if use_service_mapping:
            department = service_mapping[department] if department in service_mapping else department
        result.append(f"{start_time}-{end_time}: {department}")

    return ','.join(result)


def convert_column_to_sorted_periods_string_python(df, curr_time, time_col_name='time', col='label', check_zero=False, restrict_to_N=None):
    periods = []
    if check_zero:
        # convert nans to 0
        # df[col] = df[col].fillna(0)
        df[col] = df[col].astype(int)
        active_periods = df[df[col] == 1][time_col_name]
    else: #check for NaN
        active_periods = df[pd.isna(df[col]) == False][time_col_name]

    if active_periods.empty:
        return ""

    start_time = None
    for i, time_point in enumerate(active_periods):
        if start_time is None:
            start_time = time_point
        if i < len(active_periods) - 1 and (active_periods.iloc[i + 1] - time_point).total_seconds() // 3600 != 1:
            end_time = time_point
            periods.append((start_time, end_time))
            start_time = None
        elif i == len(active_periods) - 1:
            end_time = time_point
            periods.append((start_time, end_time))

    periods.sort(key=lambda x: x[0])
    if restrict_to_N is not None and len(periods) > restrict_to_N + 3: #space for number of past times
        times_applied_before = len(periods) - restrict_to_N
        periods = periods[-restrict_to_N:]
    else:
        times_applied_before = None

    result = []

    for start, end in periods:
        start_time = int((curr_time - start).total_seconds() // 3600) #int((curr_time - start).iloc[0].total_seconds() // 3600)
        end_time = int((curr_time - end).total_seconds() // 3600) #int((curr_time - end).iloc[0].total_seconds() // 3600)
        if start_time == end_time:
            result.append(f"{start_time}")
        else:
            result.append(f"{start_time}-{end_time}")

    if restrict_to_N is not None and times_applied_before is not None:
        result = f"{times_applied_before}x in past, current: {','.join(result)}"
    else:
        result = ','.join(result)

    return result

def convert_column_to_sorted_periods_string(df, curr_time, time_col_name='time', col='label', check_zero=False, restrict_to_N=None): #uses cython for efficiency
    # python_result = convert_column_to_sorted_periods_string_python(deepcopy(df), curr_time, time_col_name, col, check_zero, restrict_to_N)
    timestamps = df[time_col_name].values.astype('int64') // 10 ** 9  # Convert to seconds since epoch
    # Convert the 'label' column to a NumPy array of int8
    labels = df[col].fillna(-1).values.astype(np.int8)
    curr_time_unix = int(curr_time.timestamp())
    restrict_to_N = -1 if restrict_to_N is None else restrict_to_N #for cython
    cython_result = convert_column_to_sorted_periods_string_cython(timestamps, labels, curr_time_unix, check_zero, restrict_to_N)
    # assert python_result == cython_result, f"Python: {python_result} != Cython: {cython_result}"
    return cython_result

def convert_column_to_time_events_string_python(df, values, hour_time, category=None, time_col_name='charttime', use_days=False, col=None, restrict_to_N=None):
    ts_values = ""
    previous_value = None
    grouped_hours = []
    # drop NaN values
    isna = pd.isna(values)
    values = values[~isna].tolist()
    if category is not None:
        time_array = df[category][time_col_name][~isna]
    else:
        time_array = df[time_col_name][~isna]

    # try to convert to numeric
    numeric = True
    try:
        values = [float(v) for v in values]
    except ValueError:
        numeric = False
        pass

    if restrict_to_N is not None:
        if len(values) > restrict_to_N + 3:  # we use at least 3 values for calculate mean, max, min, so only apply if string would get shorter
            if numeric:
                mean_rest = round(np.mean(values[:-restrict_to_N]), 2)
                max_rest = round(np.max(values[:-restrict_to_N]), 2)
                min_rest = round(np.min(values[:-restrict_to_N]), 2)
            else:
                # get majority value
                majority = max(set(values[:-restrict_to_N]), key=values[:-restrict_to_N].count)
            values = values[-restrict_to_N:]
            time_array = time_array[-restrict_to_N:]
        else:
            mean_rest, max_rest, min_rest, majority = None, None, None, None

    last_five_index = len(values) - 5

    for i, v in enumerate(values):
        if use_days:
            hours_ago = int((hour_time - time_array.iloc[i]) / pd.Timedelta(days=1))
        else:
            if category is not None:
                hours_ago = int((hour_time - time_array.iloc[i]) / pd.Timedelta(hours=1))
            else:
                hours_ago = int((hour_time - time_array.iloc[i]) / pd.Timedelta(hours=1))

        if type(v) == float:
            # round to 1 decimal places
            v = round(v, 1)
            # cast to string
            if v.is_integer():
                v = int(v)
        if type(v) != str:
            v = str(v)

        if col == 'Estimated GFR (MDRD equation)':
            v = v.replace("Using this patient's age, gender, and ", "").replace(" value of", ":").replace("Estimated ", "").replace(
                "if African-American (mL/min/1.73 m2)", "else").replace("For comparison, ", "").replace(" (mL/min/1.73 m2)", "").replace(
                "GFR<60 = Chronic Kidney Disease, GFR<15 = Kidney Failure.", "")

        else:
            vals = v.split(',')  # multiple values in one hour
            if len(vals) > 1:
                vals = list(set([val.strip() for val in vals]))
                v = ', '.join(vals)
        if v == previous_value:
            grouped_hours.append(hours_ago)
        else:
            if previous_value is not None:
                if i - 1 < last_five_index:
                    first_hour = grouped_hours[0]
                    last_hour = grouped_hours[-1]
                    if first_hour == last_hour:
                        ts_values += f"{first_hour}: {previous_value}, "
                    else:
                        ts_values += f"{first_hour}-{last_hour}: {previous_value}, "
                else:
                    # enumerate last five times, rest are grouped
                    idxs_group = list(range(i - len(grouped_hours), i))
                    if idxs_group[0] >= last_five_index:
                        ts_values += f"{'/'.join(map(str, grouped_hours))}: {previous_value}, "
                    else:
                        # get idx of last_five_index in idxs_group
                        group_till_i = idxs_group.index(last_five_index)
                        first_hour = grouped_hours[0]
                        last_hour = grouped_hours[group_till_i - 1]
                        if first_hour == last_hour:
                            ts_values += f"{first_hour}: {previous_value}, "
                        else:
                            ts_values += f"{first_hour}-{last_hour}: {previous_value}, "
                        # enumerate last five times
                        ts_values += f"{'/'.join(map(str, grouped_hours[group_till_i:]))}: {previous_value}, "

            previous_value = v
            grouped_hours = [hours_ago]

    # Handle the last group
    if previous_value is not None:
        idxs_group = list(range(len(values) - len(grouped_hours), len(values)))
        if idxs_group[0] >= last_five_index:
            ts_values += f"{'/'.join(map(str, grouped_hours))}: {previous_value}, "
        else:
            # get idx of last_five_index in idxs_group
            group_till_i = idxs_group.index(last_five_index)
            first_hour = grouped_hours[0]
            last_hour = grouped_hours[group_till_i - 1]
            assert group_till_i - 1 >= 0
            if first_hour == last_hour:
                ts_values += f"{first_hour}: {previous_value}, "
            else:
                ts_values += f"{first_hour}-{last_hour}: {previous_value}, "
            # enumerate last five times
            ts_values += f"{'/'.join(map(str, grouped_hours[group_till_i:]))}: {previous_value}, "

    if use_days:
        ts_values = ts_values.replace(': ', ' days: ')

    if restrict_to_N is not None:
        if numeric and mean_rest is not None:
            ts_values = f"past: {max_rest}-{min_rest}, mean: {mean_rest}, recent: ".replace('.00', '').replace('.0', '') + ts_values
        elif not numeric and majority is not None:
            ts_values = f"majority: {majority}, recent: ".replace('.00', '').replace('.0', '') + ts_values

    return ts_values.strip(', ')

def convert_column_to_time_events_string(df, values, hour_time, category=None, time_col_name='charttime', use_days=False, col=None, restrict_to_N=None):
    # python_result = convert_column_to_time_events_string_python(deepcopy(df), deepcopy(values), hour_time, category, time_col_name, use_days, col, restrict_to_N)
    values = values.apply(lambda x: x if not isinstance(x, dict) else None)
    isna = pd.isna(values)
    values_np = values[~isna].to_numpy()
    numeric = True
    try:
        values_np.astype(float) # yes, we are not using the result
    except ValueError:
        numeric = False
        pass
    values_np = values_np.astype(object)
    hour_time = int(hour_time.timestamp())
    if use_days:
        time_array = df[time_col_name][~isna].values.astype('int64') // 10 ** 9  # Convert to seconds since epoch
    else:
        if category is not None:
            time_array = df[category][time_col_name][~isna].values.astype('int64') // 10 ** 9  # Convert to seconds since epoch
        else:
            time_array = df[time_col_name][~isna].values.astype('int64') // 10 ** 9  # Convert to seconds since epoch
    restrict_to_N = -1 if restrict_to_N is None else restrict_to_N #for cython
    cython_result = convert_column_to_time_events_string_cython(values_np, hour_time, time_array, use_days, numeric, col, restrict_to_N)
    # assert python_result == cython_result, f"Python: {python_result} != Cython: {cython_result}" # activate to verify equality
    return cython_result


def extract_impression(report):
    impression_names = ['impression', 'conclusion', 'findings/impression', 'impresson', 'imprression', 'imoression' 'impressoin', 'imprssion',
                        'impresion', 'imperssion', 'mpression', 'impession', 'impressions', 'impresions', 'conclusion', 'conclusions', 'impresssion']
    finding_names = ['findings', 'views', 'finding', 'findins', 'findindgs', 'findgings', 'findngs', 'findnings', 'finidngs']
    p_section = re.compile(r'\n([A-Z()/,-]+):\s', re.DOTALL)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(report, idx)
    if s:
        sections.append(report[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = report[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(report, idx_start + idx_skip)

            if s is None:
                idx_end = len(report)
            else:
                idx_end = s.start()

            sections.append(report[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(report)
        section_names.append('full report')
        section_idx.append(0)

    impression = None
    findings = None
    for i, name in enumerate(section_names):
        if name in impression_names:
            impression = sections[i]
        if name in finding_names:
            findings = sections[i]

    if impression is not None:
        return impression.strip()
    elif findings is not None:
        return findings.strip()
    else:
        # return full report
        return report.strip()
