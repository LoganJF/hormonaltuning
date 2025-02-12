import numpy as np
import pandas as pd
from glob import glob
from hormonaltuning.util import format_20min_exps_to_15mins, reorder_df, formatter_throwaway

__all__ = ['load_long_IC', 'load_spikes', 'load_burst_dataframe_from_date', 'load_and_combine_close_long_IC']

def load_long_IC(date='10-21', conditions='all', paths=None, long_burst_dur=1):
    """

    :param date:
    :param conditions:
    :param paths: str, must be of specific format for glob.glob to find; {} is replaced with the date,
                * is any character e.g. '/Users/loganfickling/Downloads/Hemo*/{}*/csvs/*spike*'
    :param long_burst_dur:
    :return:
    """
    _pn, ic = load_burst_dataframe_from_date(date=date, conditions=conditions, paths=paths)
    _pn = format_20min_exps_to_15mins(_pn)
    long_ic = _pn.loc[(_pn.Neuron == 'IC') & (_pn['Burst Duration (s)'] > long_burst_dur)]
    return long_ic


def load_spikes(date='10-21', conditions='all', paths=None):
    """

    :param date:
    :param conditions:
    :param paths: str, must be of specific format for glob.glob to find; {} is replaced with the date,
                * is any character e.g. '/Users/loganfickling/Downloads/Hemo*/{}*/csvs/*spike*'
    :return:
    """
    if paths == None:
        paths = '/Users/loganfickling/Downloads/Hemo*/{}*/csvs/*spike*'.format(date)

    folder_paths = sorted(glob(paths))
    # try:
    ic = pd.concat([pd.read_csv(x) for x in folder_paths])

    # FORMATTING spike data
    for label, cond in ic.groupby('condition'):
        ic.loc[ic['condition'] == label, 'condition'] = list(
            map(lambda s: s.replace('+', '+\n'), ic.loc[ic['condition'] == label, 'condition']))
        label = label.replace('+', '+\n')
        if 'gsif' in label:
            if '^-' not in label:
                replace = np.array(list(map(lambda s: str(s).replace('gsif', 'gsif 10^-6M'), cond.condition)))
                # np.where()
                ic.loc[ic['condition'] == label, 'condition'] = list(
                    map(lambda s: str(s).replace('gsif', 'gsif 10^-6M'), cond.condition))

    if conditions != 'all':
        ic = ic.loc[ic.condition.isin(conditions)]

    return ic

def load_burst_dataframe_from_date(date='10-21', conditions='all', paths=None):
    """

    :param date:
    :param conditions:
    :param paths: str, must be of specific format for glob.glob to find; {} is replaced with the date,
                * is any character e.g. '/Users/loganfickling/Downloads/Hemo*/{}*/csvs/*spike*'
    :return:
    """
    if paths is None:
        paths = '/Users/loganfickling/Downloads/Hemo*/{}*/csvs/*bursts*'.format(date)
    folder_paths = sorted(glob(paths))

    df = []
    for path in folder_paths:
        data = pd.read_csv(path)
        data['Condition'] = list(map(lambda s: str(s).replace(',', '\n'), data.Condition))
        data['Condition'] = list(map(lambda s: str(s).replace('+', '+\n'), data.Condition))
        data['condition'] = data['Condition']
        df.append(data)
    df = pd.concat(df)

    order_d = dict(zip(df.Condition.unique(), np.arange(df.Condition.unique().shape[0])))
    orders = list(map(lambda x: order_d[x], df.Condition))
    df['Order'] = orders

    data = formatter_throwaway(df)
    data['Duty Cycle'] = data['Burst Duration (sec)'] / data['Instantaneous Period (sec)']
    data['Interburst Duration (s)'] = data['Cycle Period'] - data['Burst Duration (s)']

    data['Condition'] = data['condition']
    order = data.Condition.unique()
    data = reorder_df(data, list_of_order=order)

    pyloric = ['IC', 'LP', 'PD', 'LPG', 'VD', 'DG', 'LG', 'AM']
    _pn = data

    # Fixing of labels...
    for label, cond in _pn.groupby('Condition'):
        if 'gsif' in label:
            if '^-' not in label:
                _pn.loc[_pn['Condition'] == label, 'Condition'] = list(
                    map(lambda s: str(s).replace('gsif', 'gsif 10^-6M'), cond.Condition))

    paths = paths.replace('bursts', 'spike')
    folder_paths = sorted(glob(paths))
    try:
        ic = pd.concat([pd.read_csv(x) for x in folder_paths])
        ic = ic[ic['neuron'] == 'IC']

        # FORMATTING spike data
        for label, cond in ic.groupby('condition'):
            ic.loc[ic['condition'] == label, 'condition'] = list(map(lambda s: s.replace('+', '+\n'), cond.condition))

        if conditions != 'all':
            _pn = _pn.loc[_pn.Condition.isin(conditions)]
            ic = ic.loc[ic.condition.isin(conditions)]
        return _pn, ic

    except Exception as e:
        print('No IC Spike files found', date)
        if conditions != 'all':
            _pn = _pn.loc[_pn.Condition.isin(conditions)]
        return _pn, None
    return


def load_and_combine_close_long_IC(date,
                                   paths=None,
                                   n_sec_cp_combine=5,
                                   long_IC_burst_dur=1,
                                   format_for_15_mins=False,
                                   conds=None):
    """"Function loads and returns long IC array used for phase plots with fixed Cycle periods to be used

    :param date: str, date the experiment was performed on
    :param paths: str, must be of specific format for glob.glob to find; {} is replaced with the date,
                * is any character e.g. '/Users/loganfickling/Downloads/Hemo*/{}*/csvs/*spike*'
    :param n_sec_cp_combine: int, default 5, time in seconds wherein bursts closer than this are combined
    :param long_IC_burst_dur: int, default 1, time in seconds to use as threshold for a long IC burst
    :param format_for_15_mins: boolean, default False, whether to format data for maximum of 15 minutes
    :param conds: array-like, condition of interest
    :return: pd.DataFrame of IC bursts with combined times

    Note: Only use one specific condition at a time! Otherwise this will be inaccurate
    """

    if conds is None:
        fed_conds = ['1-hour Fed Hemo (mackerel) +\n gsif 10^-6M',
                     '1-hour Fed Hemo (mackerel) +\n gsif 10^-6M',
                     '1-hour Fed Hemo (smelt) +\n gsif 10^-6M',
                     '1-hour Fed Hemo +\n gsif 10^-6M',
                     '1-hour Fed Hemo(mackerel) +\n gsif 10^-6M',
                     '1-hour fed hemo (mackerel) +\n gsif 10^-6M',
                     'Fed Hemo +\n gsif 10^-6M', 'Fed Hemo 0 +\n gsif 10^-6M',
                     '1-hour Fed Hemo (mackerel) + gsif 10^-6M',
                     '1-hour Fed Hemo (smelt) + gsif 10^-6M',
                     '1-hour Fed Hemo + gsif 10^-6M',
                     '1-hour Fed Hemo(mackerel) + gsif 10^-6M',
                     '1-hour fed hemo (mackerel) + gsif 10^-6M',
                     'Fed Hemo + gsif 10^-6M', 'Fed Hemo 0 + gsif 10^-6M',
                     'Fed Hemo + GSIF 10^-6M', 'Fed Hemo +\n GSIF 10^-6M',
                     '1-hour Fed Hemo (mackerel) +\n gsif 10^-6M'
                     ]
    else:
        fed_conds = conds

    _pn, ic = load_burst_dataframe_from_date(date, paths=paths)
    _pn = _pn.loc[_pn.Condition.isin(fed_conds)]
    if format_for_15_mins: # Format exps from 20 mins
        _pn = format_20min_exps_to_15mins(_pn)
    # Get long IC and calculate Cycle Period
    long_ic = _pn.loc[(_pn.Neuron == 'IC') & (_pn['Burst Duration (s)'] > long_IC_burst_dur)]
    long_ic_cycle_periods = np.append(np.diff(long_ic['Start of Burst (s)']), np.nan)
    long_ic['Cycle Period'] = long_ic_cycle_periods

    locs_valid = np.where(long_ic_cycle_periods > n_sec_cp_combine)
    editted_long_IC = long_ic.iloc[locs_valid[0]]
    editted_long_IC['Cycle Period'] = np.append(np.diff(editted_long_IC['Start of Burst (s)']), np.nan)
    return editted_long_IC  # This will still need removal of <40s

