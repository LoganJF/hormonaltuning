from scipy.stats import bootstrap
import numpy as np

def mean_diff(sample1, sample2, axis=-1):
    mean1 = np.mean(sample1, axis=axis)
    mean2 = np.mean(sample2, axis=axis)
    return mean1 - mean2


def calculate_bootstrap_diff_CI(sample1, sample2, confidence_level=.95, method='basic'):
    data = (sample1, sample2)
    res = bootstrap(data, mean_diff, method=method, confidence_level=confidence_level)  # , rng=rng)
    return res.confidence_interval


# print(mean_diff(data_uh, data_fh))
# print(res.confidence_interval)

def calculate_bootstrap_diff_CI_multi(sample1, sample2, method='basic', CIs=[.95, .99, .999, .9999, .99999]):
    _intervals = []  # {}

    for confidence_interval in CIs:
        low, high = calculate_bootstrap_diff_CI(sample1, sample2, confidence_level=confidence_interval, method='basic')
        # _intervals[confidence_interval] = (low, high)

        if (low <= 0 <= high) | (low >= 0 >= high):  # If crosses 0 not significant
            if len(_intervals) == 0:  # If doesn't pass p<.05
                return [(low, high)]
            return _intervals  # If did pass p<.05
        _intervals.append((low, high))  # Otherwise continue along
    return _intervals