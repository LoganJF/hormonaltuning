"""util.py utility functions to aid in analyses done for (Fickling et al., 2025)
author: Logan James Fickling github.com/LoganJF

TODO: format_20min_exps_to_15mins isn't strictly necessary as a function as the variable length can be used instead
"""
import numpy as np
import pandas as pd

__all__ = ['formatter_throwaway',
           'reorder_df',
           'format_20min_exps_to_15mins',
           'combine_close_long_bursts',
           'format_spikes_variable_length_exps_to_15mins',
           'remove_first_minute_spike_data',
           'return_only_desired_neurons',
           'apply_diff_append_nan_pandas',
           'inverse',
           'reformat_decile_data']

def formatter_throwaway(bursts_df):
    """Converts Logan's formatted data to match Mark Beenhakker's format for compatibility with The Crab Analyzer.

    :param bursts_df: pd.DataFrame; a dataframe analogous to Mark's style

    :return: pd.DataFrame; reformatted dataframe
    """
    bursts_df['Burst#'] = bursts_df['Burst Order']
    bursts_df['Burst Duration (sec)'] = bursts_df['Burst Duration (s)']
    bursts_df['Spike Frequency (Hz)'] = bursts_df['Spike Frequency']
    bursts_df['Date'] = bursts_df['date']
    bursts_df['Instantaneous Period (sec)'] = bursts_df['Cycle Period']
    bursts_df['Instantaneous Frequency (Hz)'] = 1./bursts_df['Cycle Period']
    return bursts_df


def reorder_df(df, list_of_order, by='Condition'):
    """Reorders a dataframe based on the specified list of conditions.

    :param df: pd.DataFrame; input dataframe
    :param list_of_order: list-like; list of values for the column to reorder by
    :param by: str; column name to reorder the dataframe by

    :return: pd.DataFrame; reordered dataframe
    """
    assert(len(list_of_order)==len(df.groupby(by)))
    return pd.concat([df[df[by]==cond]for cond in list_of_order])


def format_20min_exps_to_15mins(_pn):
    """Reformats a dataframe containing 20-minute experiments to only include the first 15 minutes of data.

    :param _pn: pd.DataFrame; input dataframe similar to Mark's output

    :return: pd.DataFrame; dataframe with only 15 minutes of data
    """
    df = _pn.groupby('Condition', as_index=False, sort=False, group_keys=False).apply(
        _format_20min_exps_to_15mins).reset_index(drop=True)
    return df


def _format_20min_exps_to_15mins(cond):
    """Helper function that truncates data for each condition to 15 minutes.

    :param cond: pd.DataFrame; data for a specific condition
    :return: pd.DataFrame; truncated dataframe

    Note: helper function, do not directly use
    """
    PD = cond.loc[cond['Neuron'] == 'PD']
    if len(PD) == 0:
        return cond
    total_time = PD.iloc[-1]['Start of Burst (s)'] - PD.iloc[0]['Start of Burst (s)']
    if total_time > 900:
        end_at = PD.iloc[0]['Start of Burst (s)'] + 900
        return cond.loc[cond['Start of Burst (s)'] < end_at]
    else:
        return cond


def combine_close_long_bursts(df, n=5):
    """Combines long IC bursts that occur within n seconds of each other.

    :param df: pd.DataFrame; input dataframe with burst data
    :param n: int; time in seconds within which to combine bursts

    :return: pd.DataFrame; dataframe with close bursts combined
    """
    diff_in_burst_starts = np.append(np.diff(df['Start of Burst (s)']), np.nan)
    locs_valid = np.where(diff_in_burst_starts < n)
    combine_with = locs_valid[0] + 1
    # This will only set the start of bursts and burst duration to be updated
    df.iloc[combine_with, df.columns.get_loc('Start of Burst (s)')] = np.array(
        df.iloc[locs_valid]['Start of Burst (s)'])
    df.iloc[combine_with, df.columns.get_loc('Burst Duration (sec)')] += np.array(
        df.iloc[locs_valid]['Burst Duration (sec)'])
    df.iloc[combine_with, df.columns.get_loc('Burst Duration (s)')] += np.array(
        df.iloc[locs_valid]['Burst Duration (s)'])
    return df.iloc[np.where(diff_in_burst_starts > n)[0]]


def format_spikes_variable_length_exps_to_15mins(df):
    """Formats spike data from experiments of variable length to ensure only the first 15 minutes
       (900 seconds) are included.

    :param df: pd.DataFrame; input dataframe with spike data

    :return: pd.DataFrame; truncated dataframe with only the first 900 seconds
    """
    _df = df.groupby('condition', as_index=False, sort=False, group_keys=False).apply(
        _format_spikes_variable_length_exps_to_15mins).reset_index(drop=True)
    return _df


def _format_spikes_variable_length_exps_to_15mins(cond):
    """Helper function that truncates spike data for each condition to 15 minutes.

    :param cond: pd.DataFrame; data for a specific condition

    :return: pd.DataFrame; truncated dataframe

    Note: helper function, do not directly use
    """
    PD = cond.loc[cond['neuron'] == 'PD']
    if len(PD) == 0:
        return cond
    total_time = PD.iloc[-1]['time'] - PD.iloc[0]['time']
    if total_time > 900:
        end_at = PD.iloc[0]['time'] + 900
        return cond.loc[cond['time'] < end_at]
    else:
        return cond


def remove_first_minute_spike_data(df):
    """Removes the first minute (60 seconds) of spike data from the dataframe.

    :param df: pd.DataFrame; input dataframe with spike data

    :return: pd.DataFrame; dataframe with the first 60 seconds of data removed
    """
    _df = df.groupby('condition', as_index=False, sort=False, group_keys=False).apply(
        _remove_first_minute_spike_data).reset_index(drop=True)
    return _df


def _remove_first_minute_spike_data(cond):
    """Helper function that removes the first minute of spike data for each condition.

    :param cond: pd.DataFrame; data for a specific condition

    :return: pd.DataFrame; truncated dataframe

    Note: helper function, do not directly use
    """
    PD = cond.loc[cond['neuron'] == 'PD']
    if len(PD) == 0:
        return cond
    remove_before = PD.iloc[0]['time'] + 60
    return cond.loc[cond['time'] >= remove_before]


def return_only_desired_neurons(df, col_name='neuron', neurons_keep=['IC', 'PD', 'LG', 'DG']):
    """Filters the dataframe to include only the specified neurons.

    :param df: pd.DataFrame; input dataframe with neuron data
    :param col_name: str; column name containing neuron data
    :param neurons_keep: list-like; list of neuron names to keep

    :return: pd.DataFrame; dataframe with only the desired neurons
    """
    return df.loc[df[col_name].isin(neurons_keep)]


def apply_diff_append_nan_pandas(dataframe, col_apply_diff_on='time', new_col_name='valid ISI'):
    """Applies the first-order difference to the specified column and appends a NaN value to the result.

    :param dataframe: pd.DataFrame; input dataframe
    :param col_apply_diff_on: str; column to apply the difference to
    :param new_col_name: str; name of the new column with the differences

    :return: pd.DataFrame; dataframe with the differences appended as a new column
    """
    diff_df = np.append(np.diff(dataframe[col_apply_diff_on]), np.nan)
    return pd.DataFrame(diff_df, columns=[new_col_name])


def inverse(df):
    """Computes the inverse of an array or dataframe (1 / value).

    :param df: array-like or pd.DataFrame; input data

    :return: array-like or pd.DataFrame; inverse of the input data
    """
    return 1 / df


def reformat_decile_data(decile_df):
    """Reformats a dataframe containing lists in cells into separate columns for each value in the list.

    :param decile_df: pd.DataFrame or pd.Series; input dataframe or series with list values

    :return: pd.DataFrame; reformatted dataframe with lists expanded into separate columns
    """
    if type(decile_df) == pd.Series:
        decile_df = decile_df.reset_index()

    df_list = [pd.DataFrame(decile_df[col].values.tolist()) for col in decile_df]

    # concatenate dataframes in list
    df_new = pd.concat(df_list, axis=1, ignore_index=True)
    df_new = df_new.rename(columns={0: 'neuron', 1: 'condition'})
    return df_new
