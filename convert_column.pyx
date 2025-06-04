from cython cimport boundscheck, wraparound
cimport numpy as np
import numpy as np  # For high-level NumPy operations

# Initialize NumPy C API
np.import_array()

@boundscheck(False)
@wraparound(False)
def convert_column_to_sorted_periods_string_cython(np.ndarray[np.int64_t] timestamps, np.ndarray[np.int8_t] labels, np.int64_t curr_time_unix, bint check_zero, int restrict_to_N=-1) -> str:
    cdef list periods = []
    cdef int i, times_applied_before = 0
    cdef int n_active_periods = 0
    cdef np.int64_t start_time = -1, end_time = -1, time_point, hours_diff

    n_total = timestamps.shape[0]

    # Filtering logic based on check_zero
    for i in range(n_total):
        if check_zero and labels[i] != 1:
            if start_time != -1:  # End the period if it was started
                end_time = timestamps[i - 1]
                periods.append((start_time, end_time))
                start_time = -1
            continue
        if not check_zero and labels[i] == -1:  # -1 indicates a former NaN
            if start_time != -1:  # End the period if it was started
                end_time = timestamps[i - 1]
                periods.append((start_time, end_time))
                start_time = -1
            continue

        # Process active periods
        time_point = timestamps[i]
        if start_time == -1:
            start_time = time_point
        if i < n_total - 1:
            hours_diff = (timestamps[i + 1] - time_point) // 3600
        else:
            hours_diff = -1

        if hours_diff != -1 and hours_diff != 1:
            end_time = time_point
            periods.append((start_time, end_time))
            start_time = -1
        elif i == n_total - 1:
            end_time = time_point
            periods.append((start_time, end_time))

    # periods.sort(key=lambda x: x[0])

    # Handle restrict_to_N logic with sentinel value
    if restrict_to_N != -1 and len(periods) > restrict_to_N + 3:
        times_applied_before = len(periods) - restrict_to_N
        periods = periods[-restrict_to_N:]

    result = []

    for start, end in periods:
        start_time_hours = (curr_time_unix - start) // 3600
        end_time_hours = (curr_time_unix - end) // 3600

        if start_time_hours == end_time_hours:
            result.append(str(start_time_hours))
        else:
            result.append("{}-{}".format(start_time_hours, end_time_hours))

    if restrict_to_N != -1 and times_applied_before > 0:
        return "{}x in past, current: {}".format(times_applied_before, ','.join(result))
    else:
        return ','.join(result)

@boundscheck(False)
@wraparound(False)
def convert_column_to_time_events_string_cython(np.ndarray[object] values, np.int64_t hour_time, np.ndarray[np.int64_t] time_array, bint use_days=False, bint numeric=True, str col=None,
                                                int restrict_to_N=-1) -> str:
    cdef list ts_values_list = []
    cdef str previous_value = None
    cdef int len_values = values.shape[0]
    cdef int i, last_five_index, hours_ago, num_grouped_hours = 0
    cdef np.ndarray[np.int32_t] grouped_hours = np.zeros(len_values, dtype=np.int32)
    cdef np.int32_t *grouped_hours_ptr = <np.int32_t *> grouped_hours.data
    cdef object majority = None
    cdef double mean_rest = float('-inf'), max_rest = float('-inf'), min_rest = float('-inf')
    cdef np.ndarray[np.float64_t] numeric_values

    # Handle restrict_to_N logic
    if restrict_to_N != -1 and len_values > restrict_to_N + 3:
        if numeric:
            numeric_values = values.astype(np.float64)
            mean_rest = round(np.mean(numeric_values[:-restrict_to_N]), 2)
            max_rest = round(np.max(numeric_values[:-restrict_to_N]), 2)
            min_rest = round(np.min(numeric_values[:-restrict_to_N]), 2)
        else:
            majority = max(set(values[:-restrict_to_N]), key=list(values[:-restrict_to_N]).count)
        values = values[-restrict_to_N:]
        time_array = time_array[-restrict_to_N:]
        len_values = values.shape[0]
    else:
        mean_rest, max_rest, min_rest, majority = float('-inf'), float('-inf'), float('-inf'), None

    last_five_index = len(values) - 5

    for i in range(len_values):
        # Calculate hours ago
        if use_days:
            hours_ago = (hour_time - time_array[i]) // (24 * 3600)  # Convert to days
        else:
            hours_ago = (hour_time - time_array[i]) // 3600  # Convert to hours

        v = values[i]

        # Handle rounding and type conversion if numeric
        if numeric:
            v = round(float(v), 1)
            v_str = str(int(v)) if v.is_integer() else str(v)
        else:
            v_str = str(v)
        # Column-specific handling for 'Estimated GFR (MDRD equation)'
        if col == 'Estimated GFR (MDRD equation)':
            v_str = v_str.replace("Using this patient's age, gender, and ", "").replace(" value of", ":").replace("Estimated ", "").replace(
                "if African-American (mL/min/1.73 m2)", "else").replace("For comparison, ", "").replace(" (mL/min/1.73 m2)", "").replace(
                "GFR<60 = Chronic Kidney Disease, GFR<15 = Kidney Failure.", "")
        else:
            if ',' in v_str:
                vals = v_str.split(',')
                vals = list(set([val.strip() for val in vals]))
                v_str = ', '.join(vals)

        # Grouping logic
        if v_str == previous_value:
            grouped_hours_ptr[num_grouped_hours] = hours_ago
            num_grouped_hours += 1
        else:
            if previous_value is not None:
                if i - 1 < last_five_index:
                    first_hour = grouped_hours_ptr[0]
                    last_hour = grouped_hours_ptr[num_grouped_hours - 1]
                    if first_hour == last_hour:
                        ts_values_list.append("{}: {}".format(first_hour, previous_value))
                    else:
                        ts_values_list.append("{}-{}: {}".format(first_hour, last_hour, previous_value))
                else:
                    group_till_i = min(last_five_index - i + num_grouped_hours, num_grouped_hours)
                    first_hour = grouped_hours_ptr[0]
                    if group_till_i <= 0:
                        ts_values_list.append("{}: {}".format('/'.join(map(str, grouped_hours[:num_grouped_hours])), previous_value))
                    else:
                        last_hour = grouped_hours_ptr[group_till_i - 1]
                        if first_hour == last_hour:
                            ts_values_list.append("{}: {}".format(first_hour, previous_value))
                        else:
                            ts_values_list.append("{}-{}: {}".format(first_hour, last_hour, previous_value))
                        ts_values_list.append("{}: {}".format('/'.join(map(str, grouped_hours[group_till_i:num_grouped_hours])), previous_value))

            # Start a new group
            previous_value = v_str
            grouped_hours_ptr[0] = hours_ago
            num_grouped_hours = 1

    # Handle the final set of grouped hours
    if previous_value is not None:
        group_till_i = min(last_five_index, len_values) - len_values + num_grouped_hours
        first_hour = grouped_hours_ptr[0]
        if group_till_i <= 0:
            ts_values_list.append("{}: {}".format('/'.join(map(str, grouped_hours[:num_grouped_hours])), previous_value))
        else:
            last_hour = grouped_hours_ptr[group_till_i - 1]
            if first_hour == last_hour:
                ts_values_list.append("{}: {}".format(first_hour, previous_value))
            else:
                ts_values_list.append("{}-{}: {}".format(first_hour, last_hour, previous_value))
            ts_values_list.append("{}: {}".format('/'.join(map(str, grouped_hours[group_till_i:num_grouped_hours])), previous_value))

    # Convert to days if needed
    if use_days:
        ts_values_list = [x.replace(': ', ' days: ') for x in ts_values_list]

    # Add summary for past events
    if restrict_to_N != -1:
        if numeric and mean_rest != float('-inf'):
            ts_values_list.insert(0, "past: {}-{}, mean: {}, recent: ".format(max_rest, min_rest, mean_rest).replace('.00', '').replace('.0', ''))
        elif not numeric and majority is not None:
            ts_values_list.insert(0, "majority: {}, recent: ".format(majority).replace('.00', '').replace('.0', ''))

    return ', '.join(ts_values_list).strip(', ').replace('recent: , ', 'recent: ')
