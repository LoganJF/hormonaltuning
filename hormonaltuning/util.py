import numpy as np
import pandas as pd

__all__ = ['formatter_throwaway',
           'reorder_df',
           'format_20min_exps_to_15mins', # There's no real need for this function given the variable one....
           'combine_close_long_bursts',
           'format_spikes_variable_length_exps_to_15mins',
           'remove_first_minute_spike_data',
           'return_only_desired_neurons',
           'apply_diff_append_nan_pandas',
           'inverse',
           'reformat_decile_data']

def formatter_throwaway(bursts_df):
    """Allows compatibility between Logan's formatted data and Mark Beenhakker's formatted data (The Crab Analyzer)

    :param bursts_df: pd.DataFrame; dataframe analogous to Mark's style
    :return:
    """
    "Changes Logan formatted code to older formats for plotting compatability"
    bursts_df['Burst#'] = bursts_df['Burst Order']
    bursts_df['Burst Duration (sec)'] = bursts_df['Burst Duration (s)']
    bursts_df['Spike Frequency (Hz)'] = bursts_df['Spike Frequency']
    bursts_df['Date'] = bursts_df['date']
    bursts_df['Instantaneous Period (sec)'] = bursts_df['Cycle Period']
    bursts_df['Instantaneous Frequency (Hz)'] = 1./bursts_df['Cycle Period']
    return bursts_df


def reorder_df(df, list_of_order, by='Condition'):
    """Reorder a dataframe by a specified order of values corresponding to column "by"

    :param df: pd.DataFrame
    :param list_of_order: list-like, list
    :param by: str, column to order by

    :return: re-ordered dataframe
    """
    assert(len(list_of_order)==len(df.groupby(by)))
    return pd.concat([df[df[by]==cond]for cond in list_of_order])

def format_20min_exps_to_15mins(_pn):
    """Reformats a dataframe of 20 minute duration to only 15 minutes

    :param _pn: pd.DataFrame similiar to Mark's output

    :return: dataframe of only 15 minutes
    """
    df = _pn.groupby('Condition', as_index=False, sort=False, group_keys=False).apply(
        _format_20min_exps_to_15mins).reset_index(drop=True)
    return df


def _format_20min_exps_to_15mins(cond):
    """Function actually doing the work for format_20min_exps_to_15mins
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
    """Combines closely occurring long IC bursts if they occur within n seconds of each other

    :param df: pd.DataFrame, must contain columns 'Start of Burst (s)', 'Burst Duration (sec)', and 'Burst Duration (s)'
    :param n: int, default 5, time in seconds to combine close bursts over

    :return: df wherein close bursts are combined
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
    """Formats spike data of variable length experiments into precisely 15 minutes (900s)

    :param df: pd.DataFrame, must contain columns 'condition', 'neuron', and 'time'
    :return: df with only first 900s
    """
    _df = df.groupby('condition', as_index=False, sort=False, group_keys=False).apply(
        _format_spikes_variable_length_exps_to_15mins).reset_index(drop=True)
    return _df


def _format_spikes_variable_length_exps_to_15mins(cond):
    """actual func doing work under the hood for format_spikes_variable_length_exps_to_15mins, do not directly use"""
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
    """Remove the first minute of spike data from the dataframe

    :param df: pd.DataFrame, must contain columns 'condition', 'neuron', and 'time'
    :return: df without the first 60s of data

    Note: Function just calls _remove_first_minute_spike_data, which does the actual work
    """
    _df = df.groupby('condition', as_index=False, sort=False, group_keys=False).apply(
        _remove_first_minute_spike_data).reset_index(drop=True)
    return _df


def _remove_first_minute_spike_data(cond):
    """actual func doing work for remove_first_minute_spike_data, do not directly use"""
    PD = cond.loc[cond['neuron'] == 'PD']
    if len(PD) == 0:
        return cond
    remove_before = PD.iloc[0]['time'] + 60

    return cond.loc[cond['time'] >= remove_before]




def return_only_desired_neurons(df,
                                col_name='neuron',
                                neurons_keep=['IC', 'PD', 'LG', 'DG']):
    """Removes all neurons from the dataframe unless they are in neurons_keep

    :param df: pd.DataFrame, must contain a column corresponding to col_name
    :param col_name: str, the column name containing neurons
    :param neurons_keep: list-like, the neurons to keep

    :return: the dataframe containing only the neurons in neurons_keep
    """
    return df.loc[df[col_name].isin(neurons_keep)]


# Utility functions
def apply_diff_append_nan_pandas(dataframe, col_apply_diff_on='time', new_col_name='valid ISI'):
    """Applies first order difference with appending of a nan value to the end, returns as dataframe

    :param dataframe: pd.DataFrame
    :param col_apply_diff_on: str, column to apply difference on
    :param new_col_name: str, name assigned to new column


    Note: apply on a dataframe already grouped by neuron and condition
    """
    isi = np.append(np.diff(dataframe[col_apply_diff_on]), np.nan)
    return pd.DataFrame(isi, columns=[new_col_name])


def inverse(df):
    """Calculate the inverse of an array/dataframe

    :param df: array-like or pd.DataFrame
    :return: 1 / df
    """
    return 1 / df

def reformat_decile_data(decile_df):
    """Reformats a dadtaframe with a cell containing lists to columns for each value of the list"""
    if type(decile_df) == pd.Series:
        decile_df = decile_df.reset_index()

    df_list = [pd.DataFrame(decile_df[col].values.tolist()) for col in decile_df]

    # concatenate dataframes in list
    df_new = pd.concat(df_list, axis=1, ignore_index=True)
    df_new = df_new.rename(columns={0: 'neuron', 1: 'condition'})
    return df_new