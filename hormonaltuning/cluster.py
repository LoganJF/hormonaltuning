import pandas as pd
import numpy as np
from hormonaltuning.io import (load_and_combine_close_long_IC,
                               load_spikes)

from hormonaltuning.util import (remove_first_minute_spike_data,
                                 format_spikes_variable_length_exps_to_15mins,
                                 apply_diff_append_nan_pandas,
                                 inverse,
                                 return_only_desired_neurons,
                                 reformat_decile_data
                                 )

from hormonaltuning.extractfeatures import (pad_end_long_IC,
                                            create_phase_space_arr_from_long_IC,
                                            split_spike_data_into_list_of_segments,
                                            check_all_neurons_included_add_filler_if_not,
                                            pad_spike_data_to_compensate_for_terminal_silence,
                                            calculate_burstiness,
                                            calculate_long_IC_phase_each_spike
                                            )

# TODO: Add in spike_df argument so they can skip loading LF uses
def feature_extraction_preprocessing_pipeline(date, cond, paths=None,
                                              filler_value=-100,
                                              segment_duration_seconds=120,
                                              thres_max_CP_long_IC=40,
                                              thres_max_time_from_nearest_phase=5,
                                              n_sec_cp_combine=5,
                                              long_IC_burst_dur=1,
                                              neurons_keep=['IC', 'PD', 'AM', 'LG', 'DG'],
                                              phase_bins=100, verbose=True, ):
    """

    :param date:
    :param cond:
    :param paths:
    :param filler_value:
    :param segment_duration_seconds:
    :param thres_max_CP_long_IC:
    :param thres_max_time_from_nearest_phase:
    :param n_sec_cp_combine:
    :param long_IC_burst_dur:
    :param neurons_keep:
    :param phase_bins:
    :param verbose:
    :return:
    """
    # First, load spike data, remove first minute, only load relevant neurons and assure it's only 900s of data
    spike_df = load_spikes(date, paths=paths)
    spike_df = spike_df.loc[spike_df.condition == cond]
    spike_df = format_spikes_variable_length_exps_to_15mins(spike_df)
    spike_df = remove_first_minute_spike_data(spike_df)
    spike_df = return_only_desired_neurons(spike_df, col_name='neuron', neurons_keep=neurons_keep)

    # Now load the IC data combining periods where >1s bursts are <5s apart
    long_IC_editted = load_and_combine_close_long_IC(date,
                                                     paths=paths,
                                                     n_sec_cp_combine=n_sec_cp_combine,
                                                     long_IC_burst_dur=long_IC_burst_dur,
                                                     format_for_15_mins=False,
                                                     conds=[cond])
    has_long_IC = True
    # If all values are nan, there were no long IC
    if all(pd.isna(long_IC_editted.iloc[0])):
        if verbose:
            print('no long IC detected, setting phase with filler values', flush=True)
        has_long_IC = False
        spike_df['phase bin'] = filler_value

    if has_long_IC:  # This will be done in cases where it's valid
        long_IC_editted = pad_end_long_IC(long_IC_editted)
        phase_arr = create_phase_space_arr_from_long_IC(long_IC_editted, phase_space=phase_bins)

        # Determine phase for each spike
        spike_df = calculate_long_IC_phase_each_spike(spike_df, phase_arr, long_IC_editted,
                                                      filler_value=filler_value,
                                                      phase_bins=phase_bins,
                                                      thres_max_time_from_nearest=thres_max_time_from_nearest_phase,
                                                      thres_max_CP_long_IC=thres_max_CP_long_IC)
    if not has_long_IC:
        spike_df['phase bin'] = filler_value

    # Create segments by splitting the data into chunks of desired duration
    segment_starts, segments = split_spike_data_into_list_of_segments(spike_df,
                                                                      segment_duration_seconds=segment_duration_seconds)

    segment_features_df = []
    for index, segment_df in enumerate(segments):
        try:
            _segment_df = extract_features_from_segment(segment_df=segment_df,
                                                        segment_start=segment_starts[index],
                                                        long_IC_editted=long_IC_editted,
                                                        segment_duration_seconds=segment_duration_seconds,
                                                        neurons_keep=neurons_keep, )
            segment_features_df.append(_segment_df)
        except Exception as e:
            print(e, flush=True)
            print('segment {} failed'.format(index), flush=True)

    segment_features_df = pd.concat(segment_features_df)
    segment_features_df = segment_features_df.reset_index().rename(columns={'index': 'ID'})
    ids = ['{} {} segment {}'.format(date, cond, i) for i in np.arange(len(segment_features_df))]
    segment_features_df['ID'] = ids

    return segment_features_df


def extract_features_from_segment(segment_df,
                                  segment_start,
                                  long_IC_editted,
                                  segment_duration_seconds,
                                  neurons_keep):
    """Extracts features used to cluster from a segment

    :param segment_df:
    :param segment_start:
    :param long_IC_editted:
    :param segment_duration_seconds:
    :param neurons_keep:
    :return:
    """
    df = segment_df

    start = segment_start  # segment_starts[0]
    # Ensure that neuron recordings with no spikes are handled correctly
    df = check_all_neurons_included_add_filler_if_not(df, neuron_list=neurons_keep)

    # Calculate first order difference of spikes (ISI)
    first_order_isi = df.groupby(['neuron', 'condition'], as_index=True, sort=False).apply(apply_diff_append_nan_pandas)
    first_order_isi['time'] = first_order_isi['valid ISI']
    # Determine maximum for first order ISI
    max_ISI_1 = first_order_isi.groupby(['neuron', 'condition'], as_index=True, sort=False).max()

    # Use this to pad terminal silence, then redo each
    df = pad_spike_data_to_compensate_for_terminal_silence(df, max_ISI_1, segment_starting_time=start)

    # Redoing with pad and calculation of second order
    first_order_isi = df.groupby(['neuron', 'condition'], as_index=True, sort=False).apply(apply_diff_append_nan_pandas)
    first_order_isi['time'] = first_order_isi['valid ISI']
    second_order_isi = first_order_isi.groupby(['neuron', 'condition'], as_index=True, sort=False).apply(
        apply_diff_append_nan_pandas)
    second_order_isi['time'] = second_order_isi['valid ISI']

    # Determine maximum for first order and second order ISI
    max_ISI_1 = first_order_isi.groupby(['neuron', 'condition'], as_index=True, sort=False).max()
    max_ISI_2 = second_order_isi.groupby(['neuron', 'condition'], as_index=True, sort=False).max()

    # -----------> IMPORTANT FEATURE FOR MODEL: ratio second order to first order diff spike time
    # Use this to calculate feature of the ratio of the second to frist order differential
    ratio_second_order_first_order_isi = max_ISI_2 / max_ISI_1  # This is a feature to use in the model!!!

    # Calculations of the ratio of the largest to second largest spike
    largest_second_largest_isi = []
    for n, isi_df in first_order_isi.groupby(['neuron', 'condition'], as_index=True, sort=False):
        temp_df = isi_df.copy(deep=True)
        temp_df['valid ISI'] = temp_df['valid ISI'].astype(float)
        largest_second_largest_isi.append(temp_df.nlargest(n=2, columns='valid ISI'))
    largest_second_largest_isi = pd.concat(largest_second_largest_isi)

    ISI_maximum = largest_second_largest_isi.iloc[::2].reset_index(drop=False)
    ISI_maximum = ISI_maximum.drop(columns='level_2')
    ISI_second_maximum = largest_second_largest_isi.iloc[1::2].reset_index(
        drop=False)  # Pick every other starting with 1st index
    ISI_second_maximum = ISI_second_maximum.drop(columns='level_2')

    ratio_largest_second_largest_isi = ISI_maximum.copy()  # At first make a copy only contains id columns
    ratio_largest_second_largest_isi = ratio_largest_second_largest_isi[['neuron', 'condition']]
    # -----------> IMPORTANT FEATURE FOR MODEL: ratio largest second largest ISI
    ratio_largest_second_largest_isi['valid ISI'] = ISI_maximum['valid ISI'] / ISI_second_maximum[
        'valid ISI']  # Now set the values for the feature

    # -----------> IMPORTANT FEATURE FOR MODEL: ISI Deciles
    desired_deciles = np.arange(0, 110, 10)  # Goes up to 110 b/c it's exclusive, ending at 100

    _first_order_isi = first_order_isi.dropna()
    isi_deciles = _first_order_isi.groupby(['neuron', 'condition'], sort=False).apply(np.percentile, desired_deciles)
    isi_deciles = reformat_decile_data(isi_deciles)

    # Calculation of Firing rate
    firing_rate = first_order_isi.loc[first_order_isi['valid ISI'] < .25]
    firing_rate = firing_rate.groupby(['neuron', 'condition'], as_index=True, sort=False, group_keys=True).apply(
        inverse)
    firing_rate = firing_rate.groupby(['neuron', 'condition'], as_index=True, sort=False).mean()
    firing_rate = firing_rate.rename(columns={'valid ISI': 'mean FR'})

    # -----------> IMPORTANT FEATURE FOR MODEL: Firing Rate
    firing_rate = firing_rate.drop(columns=['time'])

    # -----------> IMPORTANT FEATURE FOR MODEL: burstiness
    burstiness = df.groupby(['neuron', 'condition'], as_index=True, sort=False).apply(calculate_burstiness)

    # -----------> IMPORTANT FEATURE FOR MODEL: phase deciles
    phase_deciles = df.dropna(axis=1).groupby(['neuron', 'condition'], sort=False, as_index=True)['phase bin'].apply(
        np.percentile, desired_deciles)
    phase_deciles = reformat_decile_data(phase_deciles)

    # -----------> IMPORTANT FEATURE FOR MODEL: # Long IC in segment
    number_long_IC_in_segment = long_IC_editted.loc[(long_IC_editted['Start of Burst (s)'] >= start)
                                                    & (long_IC_editted[
                                                           'Start of Burst (s)'] < start + segment_duration_seconds)]
    number_long_IC_in_segment = len(number_long_IC_in_segment)
    segment_features_df = concat_to_single_feature_dataframe_per_segment(isi_deciles=isi_deciles,
                                                                         phase_deciles=phase_deciles,
                                                                         firing_rate=firing_rate,
                                                                         ratio_second_order_first_order_isi=ratio_second_order_first_order_isi,
                                                                         ratio_largest_second_largest_isi=ratio_largest_second_largest_isi,
                                                                         burstiness=burstiness,
                                                                         number_long_IC_in_segment=number_long_IC_in_segment,
                                                                         neurons_keep=neurons_keep)

    return segment_features_df


def concat_to_single_feature_dataframe_per_segment(isi_deciles,
                                                   phase_deciles,
                                                   firing_rate,
                                                   ratio_second_order_first_order_isi,
                                                   ratio_largest_second_largest_isi,
                                                   burstiness,
                                                   number_long_IC_in_segment,
                                                   neurons_keep):
    """

    :param isi_deciles:
    :param phase_deciles:
    :param firing_rate:
    :param ratio_second_order_first_order_isi:
    :param ratio_largest_second_largest_isi:
    :param burstiness:
    :param number_long_IC_in_segment:
    :param neurons_keep:
    :return:
    """
    cols = []
    _data = []

    for neuron in neurons_keep:
        col0 = ['{} ISI Decile '.format(neuron) + str(x) for x in np.arange(0, 11)]
        col1 = ['{} Phase Decile '.format(neuron) + str(x) for x in np.arange(0, 11)]
        col2 = ['{} Additional feature '.format(neuron) + str(x) for x in np.arange(0, 4)]
        cols.extend(col0)
        cols.extend(col1)
        cols.extend(col2)

        try:
            col0_data = list(
                isi_deciles.loc[isi_deciles['neuron'] == neuron].values[0, 2:])  # First 2 are IDs we discard
        except IndexError as e:
            print(e, flush=True)
            print('failed due to neuron {}'.format(neuron), flush=True)
            return None

        col1_data = list(phase_deciles.loc[phase_deciles['neuron'] == neuron].values[0, 2:])
        # ----- > This is all really a stupid way of formatting it oh well, fix later
        # Formatting of additional features
        fr = firing_rate.reset_index()
        fr = fr.loc[fr['neuron'] == neuron]

        try:
            fr = fr.iloc[0]['mean FR']
        except IndexError:  # This will occur when there isn't an element to reference
            print('No firing rate, setting to 0: ', neuron)
            fr = 0

        ratio_a = ratio_second_order_first_order_isi.reset_index()
        ratio_a = ratio_a.loc[ratio_a['neuron'] == neuron]
        ratio_a = ratio_a.iloc[0]['valid ISI']
        if np.isnan(ratio_a):
            ratio_a = 0

        ratio_b = ratio_largest_second_largest_isi.reset_index()
        ratio_b = ratio_b.loc[ratio_b['neuron'] == neuron]
        ratio_b = ratio_b.iloc[0]['valid ISI']
        if np.isnan(ratio_b):  # Shouldn't get the index error
            ratio_b = 0

        _burstiness = burstiness.reset_index()
        _burstiness = _burstiness.loc[_burstiness.neuron == neuron]
        _burstiness = _burstiness.iloc[0][0]

        segment_data_features = [*col0_data, *col1_data, fr, ratio_a, ratio_b, _burstiness]
        _data.extend(segment_data_features)
    _data.append(number_long_IC_in_segment)

    cols.append('# Long IC Bursts')

    segment_features_df = pd.DataFrame(_data, index=cols).T
    return segment_features_df
