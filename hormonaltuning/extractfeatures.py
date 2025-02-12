import numpy as np
import pandas as pd
from scipy.stats import zscore

__all__ = ['get_ISI_per_time', 'calculate_percentile', 'split_spike_data_into_list_of_segments',
           'pad_spike_data_to_compensate_for_terminal_silence', 'check_all_neurons_included_add_filler_if_not',
           'calculate_burstiness', 'create_phase_space_arr_from_long_IC', 'pad_end_long_IC',
           'calculate_long_IC_phase_each_spike',]


def get_ISI_per_time(df, time_range):
    """

    :param df:
    :param time_range:
    :return:
    """
    binned_ISI = []
    neuron_name = df.neuron.iloc[0]
    for i, start in enumerate(time_range):
        try:
            end = time_range[i + 1]
        except IndexError:
            break

        sliced = df.loc[(df['time'] > start) & (df['time'] < end)]
        isi = np.diff(sliced['time'])
        time_of_spikes = np.array(sliced['time'])[:-1]
        _df = pd.DataFrame(isi, columns=['isi'])
        _df['neuron'] = neuron_name
        _df['time bin'] = i
        _df['time of spike'] = time_of_spikes

        binned_ISI.append(_df)

    return pd.concat(binned_ISI)


def calculate_percentile(df):
    """Calculates percentiles of a dataframe

    :param df: pd.DataFrame, must contain columns 'condition', 'neuron', 'isi', and 'time bin'
    :return: percentiles_df, pd.DataFrame, percentiles of inputted values
    """
    get_percentiles_for = np.arange(0, 100, 3)
    percentiles = np.percentile(a=df['isi'], q=get_percentiles_for)
    percentiles_df = pd.DataFrame(percentiles, columns=['isi percentiles'])
    percentiles_df['neuron'] = df.neuron.iloc[0]
    percentiles_df['time bin'] = df['time bin'].iloc[0]
    percentiles_df['condition'] = df['condition'].iloc[0]
    percentiles_df['isi normed'] = zscore(percentiles)
    return percentiles_df


def split_spike_data_into_list_of_segments(df,
                                           segment_duration_seconds=120,
                                           exp_length=900,
                                           first_min_remove=True,
                                           return_starting_times=True):
    """Takes spike data and splits it into a list of segments of specified duration

    :param df:
    :param segment_duration_seconds:
    :param exp_length:
    :param first_min_remove:
    :param return_starting_times:
    :return:
    """
    assert (len(df.condition.unique()) == 1)
    ending_time = exp_length
    if first_min_remove:
        ending_time -= 60
    PD = df.loc[df['neuron'] == 'PD']
    starting_time = PD.iloc[0]['time']
    slices = np.arange(starting_time, starting_time + ending_time, step=segment_duration_seconds)
    _sliced_df = []

    for i, s in enumerate(slices):
        if i + 1 == len(slices):  # We don't want to go past the final point!
            break
        sliced_df = df.loc[(df['time'] >= slices[i]) & (df['time'] <= slices[i + 1])]
        _sliced_df.append(sliced_df)

    if return_starting_times:
        return slices, _sliced_df
    else:
        return _sliced_df


def pad_spike_data_to_compensate_for_terminal_silence(segmented_spike_data,
                                                      maximum_ISI_df,
                                                      segment_starting_time,
                                                      thres_rel_max_ISI=2,
                                                      segment_duration=120, ):
    """Compensates for terminal silence in neuron firing by padding the inter-spike intervals (ISIs) with fake spikes.


    :param segmented_spike_data:
    :param maximum_ISI_df:
    :param segment_starting_time:
    :param thres_rel_max_ISI:
    :param segment_duration:
    :return: Updated spike dataframe with compensated ISIs.
    """

    max_isi_per_neuron = maximum_ISI_df.reset_index()
    start = segment_starting_time
    end = start + segment_duration

    df_with_pads = []

    for (neuron, cond), _df in segmented_spike_data.groupby(['neuron', 'condition'], as_index=True, sort=False):

        max_isi_of_neuron = \
        max_isi_per_neuron.loc[(max_isi_per_neuron.neuron == neuron) & (max_isi_per_neuron.condition == cond)][
            'valid ISI']
        minimum_time_of_first_spike = _df['time'].min()
        maximum_time_of_last_spike = _df['time'].max()

        if (minimum_time_of_first_spike - start) > thres_rel_max_ISI * float(max_isi_of_neuron):
            print('needs add start')
            append_to_spikes = _df.iloc[:1].copy(deep=True)
            append_to_spikes['time'] = start
            df_with_pads.append(append_to_spikes)

        df_with_pads.append(_df)

        if (end - maximum_time_of_last_spike) > thres_rel_max_ISI * float(max_isi_of_neuron):
            # Need to append at end
            append_to_spikes = _df.iloc[:1].copy(deep=True)
            append_to_spikes['time'] = end
            print('needs add end')
            df_with_pads.append(append_to_spikes)

    df_with_pads = pd.concat(df_with_pads)
    return df_with_pads


def check_all_neurons_included_add_filler_if_not(df, neuron_list=['IC', 'PD', 'LG', 'DG']):
    """Checks that are neurons in the dataframe have activity, and adds minimum filler values if not

    :param df:
    :param neuron_list:
    :return:

    Note: In situations where there are no spikes or 1 spike, insert filler spikes at minimum and maximum point
    """
    _df = []

    for neuron in neuron_list:
        df_neuron = df.loc[df.neuron == neuron]
        if len(df_neuron) <= 1:
            # In situations where there are no spikes or 1 spike, insert filler spikes at minimum and maximum point
            make_shift_arr = pd.concat([pd.DataFrame(df.min()).T, pd.DataFrame(df.max()).T])
            make_shift_arr['neuron'] = neuron
            _df.append(make_shift_arr)

        if len(df_neuron) >= 1:
            _df.append(df_neuron)

    _df = pd.concat(_df)
    return _df


def calculate_burstiness(df, maximum_value=50):
    """Metric to capture how burst like a train of spikes is

    :param df:
    :param maximum_value:
    :return:

    Note: Make sure to use groupby first
    """
    isi = np.append(np.diff(df['time']), -999)
    sorted_isi = np.sort(isi)
    isi_gaps = np.append(np.diff(sorted_isi), -999)
    max_gap, pos = np.max(isi_gaps), np.argmax(isi_gaps)
    burstiness = max_gap / isi[pos]

    if burstiness > maximum_value:
        burstiness = maximum_value
    return burstiness



def create_phase_space_arr_from_long_IC(combined_long_IC_bursts, phase_space=100):
    """

    :param combined_long_IC_bursts:
    :param phase_space:
    :return:
    """
    data = combined_long_IC_bursts

    creator_starts = data['Start of Burst (s)']
    phase_dictionary = {}
    for i, start_value in enumerate(creator_starts):
        try:
            creator_next_burst = creator_starts.iloc[i + 1]
        except IndexError:
            continue
        step = (creator_next_burst - start_value) / phase_space
        arr = np.arange(start_value, creator_next_burst, step=step)

        if len(arr) != phase_space:
            arr = arr[:phase_space]
        assert (len(arr) == phase_space)
        phase_dictionary[i] = arr

    phase_arr = np.concatenate(list(phase_dictionary.values()), dtype='object')
    return phase_arr


def pad_end_long_IC(long_IC_burst):
    """Makes the final long IC burst of an experiment have a cycle period equal to the mean
    this allows it to still be treated as sensible phase data before creating areas where it will be
    undefined

    :param long_IC_burst:
    :return:
    """
    mean_cp = long_IC_burst['Cycle Period'].mean()
    if pd.isna(long_IC_burst.iloc[-1]['Cycle Period']):

        last_row = long_IC_burst.iloc[-1:].copy(deep=True)
        df = long_IC_burst.copy()
        df.iloc[-1, df.columns.get_loc('Cycle Period')] = mean_cp
        time_of_pseudo_burst = float(df.iloc[-1]['Start of Burst (s)'])
        time_of_pseudo_burst += mean_cp

        last_row['Start of Burst (s)'] = time_of_pseudo_burst
        last_row['Neuron'] = 'IC'
        df = pd.concat([df, last_row])

        return df
    else:
        return long_IC_burst


from stns import find_nearest


def calculate_long_IC_phase_each_spike(spike_df, phase_arr, long_IC_editted,
                                       phase_bins=100,
                                       thres_max_time_from_nearest=5,
                                       thres_max_CP_long_IC=40,
                                       filler_value=-100):
    """Utility function which calculates for each spike, the closest phase bin relative to long IC

    :param spike_df:
    :param phase_arr:
    :param long_IC_editted:
    :param phase_bins:
    :param thres_max_time_from_nearest:
    :param thres_max_CP_long_IC:
    :param filler_value:
    :return:
    """

    spike_times = spike_df['time'].values
    nearest_index = np.array([find_nearest(array=phase_arr, value=x) for x in spike_times])
    nearest_time = phase_arr[nearest_index]
    # Calculate absolute difference in time between nearest phase bin and time of spike
    difference_in_closeness = np.abs(spike_times - np.array(nearest_time))
    locs_replace_with_filler = np.where(difference_in_closeness > thres_max_time_from_nearest)
    nearest_long_IC_phase_bin = nearest_index % phase_bins  # This allows each trial to be 0-99

    # First, any value above the threshold is set to be -100
    if len(locs_replace_with_filler[0]) > 0:
        nearest_long_IC_phase_bin[locs_replace_with_filler] = filler_value

    # Next, any value where cycle period >40 is set to be invalid
    locs_cycle_period_slow = np.where(long_IC_editted['Cycle Period'] > thres_max_CP_long_IC)
    if len(locs_cycle_period_slow[0]) > 0:
        bad_starts = locs_cycle_period_slow[0] * phase_bins
        for remove_start in bad_starts:
            remove_end = remove_start + phase_bins
            locs_fix = np.where((nearest_index >= remove_start) & (nearest_index < remove_end))
            nearest_long_IC_phase_bin[locs_fix] = filler_value

    spike_df['phase bin'] = nearest_long_IC_phase_bin
    return spike_df


