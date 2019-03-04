import numpy as np


def compute_best_feature(x, y, weights, classes, criterion='gini'):
    """
    Compute best feature to perform binary split according to criterion on a
    given population (x, y).

    Parameters
    ----------
    x: np.array
        Features.
    y: np.array
        Targets.
    classes: np.ndarray
        Array of different values of target in train dataset.
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
        best_metric, best_value = compute_best_split(feature, y,
                                                     weights, classes,
                                                     metric)
        feature_best_split_metrics.append(best_metric)
        feature_best_split_value.append(best_value)

    # Best value is minimal value
    best_feature = np.argmin(feature_best_split_metrics)
    best_metric = feature_best_split_metrics[best_feature]
    best_value = feature_best_split_value[best_feature]

    return best_feature, best_metric, best_value


def compute_best_split(feature, y, weights, classes, metric):
    """
    Compute best value to perform split on given feature for given metric.

    Parameters
    ----------
    feature: 1D-np.array
        Feature.
    y: np.array
        Target.
    classes: np.ndarray
        Array of different values of target in train dataset.
    metric: func
        Criterion for defining best split. Can be either gini or entropy.
    """
    count = 0
    metrics_list = []
    split_values = []

    # Case : continuous
    for value in np.sort(feature)[:-1]:
        left_idx, right_idx = split(feature, value)
        metrics_list.append(split_metric(left_idx, right_idx,
                                         y, weights,
                                         classes, metric))
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
    return np.argwhere(x <= value).flatten(), np.argwhere(x > value).flatten()


def estimate_probas(y, weights, classes):
    """
    Compute empirical probabilities for classes given y taking into account
    sample weights.
    """
    return [np.dot(weights, (y == tar)) / np.sum(weights) for tar in classes]


def gini(y, weights, classes):
    """
    Compute gini of target vector y.
    """
    probas = estimate_probas(y, weights, classes)
    return np.sum(np.array(probas) * (1 - np.array(probas)))


def entropy(y, weights, classes):
    """
    Compute entropy of target vector y.
    """
    epsilon = 1e-6
    probas = estimate_probas(y, weights, classes)
    return - np.sum(np.array(probas) * np.log(np.array(probas) + epsilon))


def split_metric(left_idx, right_idx, y, weights, classes, metric):
    """
    Compute metric for given split, taking weighted average over the two
    populations.
    """
    left_y, right_y = y[left_idx], y[right_idx]
    left_weights, right_weights = weights[left_idx], weights[right_idx]
    return (np.sum(left_weights) * metric(left_y,
                                          left_weights,
                                          classes)
            + np.sum(right_weights) * metric(right_y,
                                             right_weights,
                                             classes)) \
        / np.sum(weights)
