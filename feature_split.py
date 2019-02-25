import numpy as np


def compute_best_feature(x, y, criterion='gini'):
    """
    Compute best feature to perform binary split according to criterion on a
    given population (x, y).

    Parameters
    ----------
    x: np.array
        Features.
    y: np.array
        Targets.
    criterion: str
        Criterion for defining best split. Can be either 'entropy' or 'gini'.
    """
    if criterion == 'gini':
        metric = gini
    if criterion == 'entropy':
        metric = entropy

    feature_best_split_metrics = []
    feature_best_split_value = []
    for feature in x.T:
        best_metric, best_value = compute_best_split(feature, y, metric)
        feature_best_split_metrics.append(best_metric)
        feature_best_split_value.append(best_value)

    best_feature = np.argmin(feature_best_split_metrics)
    best_metric = feature_best_split_metrics[best_feature]
    best_value = feature_best_split_value[best_feature]

    return best_feature, best_metric, best_value


def compute_best_split(feature, y, metric):
    """
    Compute best value to perform split on given feature for given metric.

    Parameters
    ----------
    feature: 1D-np.array
        Feature.
    y: np.array
        Target.
    metric: func
        Criterion for defining best split. Can be either gini or entropy.
    """
    count = 0
    metrics_list = []
    split_values = []

    # Case : continuous
    for value in sorted(feature):
        left_split, right_split = split(feature, value)
        metrics_list.append(split_metric(y[left_split].flatten(),
                                         y[right_split].flatten(),
                                         metric))
        split_values.append(value)
        count += 1

    # TODO : categorical

    # Compute min and return value
    idx_max = np.argmin(np.array(metrics_list))

    best_metric, best_value = metrics_list[idx_max], split_values[idx_max]

    return best_metric, best_value


def split(x, value):
    """
    Return indexes of samples in the two populations.
    """
    return np.argwhere(x <= value), np.argwhere(x > value)


def gini(y):
    """
    Compute gini of target vector y.
    """
    probas = np.bincount(y) * 1.0 / len(y)
    return sum([p * (1-p) for p in probas])


def entropy(y):
    """
    Compute entropy of target vector y.
    """
    epsilon = 1e-6
    probas = np.bincount(y) * 1.0 / len(y)
    return -sum([p * np.log(p + epsilon) for p in probas])


def split_metric(left_split, right_split, metric):
    """
    Compute metric for given split, taking average over the two populations.
    """
    n = len(left_split) + len(right_split)
    return (len(left_split) * metric(left_split) + len(right_split)  * metric(right_split)) / n
