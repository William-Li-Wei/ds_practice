from typing import Union
from numbers import Number

import numpy as np
import pandas as pd
from scipy import stats


def get_confidence_interval(
    samples: Union[np.ndarray, pd.Series],
    confidence_level: float,
    critical_value: str = "z",
    tail: str = "both",
    population_std: float = None
):
    """
    Return the confidence interval accordingly

    if population_std is given, use z-value,
    else estimate population_std by sample std, then use the specified critical value

    Args:
        samples(np.ndarray): measurements of samples
        confidence_level(float): 1 - alpha
        critical_value(str): default to "z", z-value, "t" for t-value
        tail(str): default to "both". "left" for single left tail or "right" for single right tail
        population_std(float): Standard deviation of the population. default to None

    Returns:
    """

    if isinstance(samples, pd.Series):
        samples = samples.to_numpy()
    elif not isinstance(samples, np.ndarray):
        raise Exception("Invalid parameter, samples should be either a numpy array or a pandas Series")

    if not isinstance(confidence_level, Number) or confidence_level > 1 or confidence_level < 0:
        raise Exception("Invalid parameter, confidence_level should be a number between [0, 1]")

    if critical_value not in ["z", "t"]:
        raise Exception("Invalid parameter, critical_value should be one of [\"z\", \"t\"]")

    if tail not in ["left", "both", "right"]:
        raise Exception("Invalid parameter, confidence_level should be one of [\"left\", \"both\", \"right\"]")

    if population_std is not None and (
        not isinstance(population_std, Number) or population_std < 0
    ):
        raise Exception("Invalid parameter, population_std should be a positive number")

    alpha = 1 - confidence_level
    n = samples.shape[0]
    sample_mean = np.mean(samples)

    # use population_std or sample_std
    if population_std is not None:
        std = population_std
        critical_value = "z"
    else:
        std = np.std(samples, ddof=1)

    # get critical value accordingly
    if critical_value is "z":
        if tail is "both":
            critical_value = stats.norm.ppf(1 - alpha / 2)
        else:
            critical_value = stats.norm.ppf(1 - alpha)
    if critical_value is "t":
        if tail is "both":
            critical_value = stats.t.ppf(1 - alpha / 2, df=n-1)
        else:
            critical_value = stats.t.ppf(1 - alpha, df=n-1)

    # get lower and upper
    lower = sample_mean - critical_value * std / np.sqrt(n)
    upper = sample_mean + critical_value * std / np.sqrt(n)
    if tail == "right":
        upper = np.Infinity
    if tail == "left":
        lower = -np.Infinity

    return lower, upper
