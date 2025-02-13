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

def calculate_bootstrap_diff_CI_multi(sample1, sample2, method='basic', CIs=[.95, .99, .999, .9999, .99999]):
    _intervals = []  # {}

    for confidence_interval in CIs:
        low, high = calculate_bootstrap_diff_CI(sample1, sample2, confidence_level=confidence_interval, method=method)

        if (low <= 0 <= high) | (low >= 0 >= high):  # If it crosses 0 and thus not significant
            if len(_intervals) == 0:  # If it doesn't pass p<.05
                return [(low, high)]
            return _intervals  # If it did pass p<.05
        _intervals.append((low, high))  # Otherwise continue along
    return _intervals


from scipy.stats import bootstrap
import numpy as np


def mean_diff(sample1, sample2, axis=-1):
    """
    Calculate the difference in means between two samples.

    :param sample1: np.ndarray, The first sample to calculate the mean from.
    :param sample2: np.ndarray, The second sample to calculate the mean from.
    :param axis: int, The axis along which to calculate the mean (default is -1 for the last axis).

    :return: np.ndarray or float, The difference between the means of sample1 and sample2.
    """
    mean1 = np.mean(sample1, axis=axis)
    mean2 = np.mean(sample2, axis=axis)
    return mean1 - mean2


def calculate_bootstrap_diff_CI(sample1, sample2, confidence_level=.95, method='basic'):
    """
    Calculate the confidence interval for the difference in means between two samples using the bootstrap method.

    :param sample1: np.ndarray, The first sample.
    :param sample2: np.ndarray, The second sample.
    :param confidence_level: float, The confidence level for the interval (default is 0.95).
    :param method: str, The method used for bootstrap confidence interval calculation (default is 'basic').

    :return: scipy ConfidenceInterval, The confidence interval for the difference in means.
    """
    data = (sample1, sample2)
    res = bootstrap(data, mean_diff, method=method, confidence_level=confidence_level)
    return res.confidence_interval


def calculate_bootstrap_diff_CI_multi(sample1, sample2, method='basic', CIs=[.95, .99, .999, .9999, .99999]):
    """
    Calculate bootstrap confidence intervals for the difference in means at multiple confidence levels,
    and return the highest interval where the difference is significant (doesn't cross zero).

    :param sample1: np.ndarray, The first sample.
    :param sample2: np.ndarray, The second sample.
    :param method: str, The method for bootstrap confidence interval calculation (default is 'basic').
    :param CIs: list, A list of confidence levels to calculate (default is [.95, .99, .999, .9999, .99999]).

    :return: list of tuples, List of confidence intervals that do not cross zero, indicating statistical significance.
    """
    confidence_intervals = []  # List to store significant intervals

    for confidence_interval in CIs:
        low, high = calculate_bootstrap_diff_CI(sample1, sample2, confidence_level=confidence_interval, method=method)

        # Check if the confidence interval crosses zero (i.e., not statistically significant)
        if (low <= 0 <= high) | (low >= 0 >= high):
            if len(confidence_intervals) == 0:  # If no significant intervals have been found, return the first one
                return [(low, high)]
            return confidence_intervals  # # occurs if it did pass p<.05, return previously found significant intervals

        # Add the significant interval to the list
        confidence_intervals.append((low, high))

    return confidence_intervals
