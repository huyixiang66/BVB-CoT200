"""VSI-Bench numeric metric; inputs must already be parsed numeric values."""


def vsi_mean_relative_accuracy(prediction, target):
    if target <= 0:
        raise ValueError('target must be positive')
    error = abs(prediction-target)/target
    thresholds = [0.5 + i*(0.45/10.0) for i in range(11)]
    return sum(error <= 1.0-threshold for threshold in thresholds)/len(thresholds)
