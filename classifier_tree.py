import numpy as np

from feature_split import estimate_probas, compute_best_feature, split


class Binary_Tree(object):
    """
    Class defining the recursive binary tree data structure.
    """
    def __init__(self, node=None, probas=None):
        """
        Initialize tree with node and probas arguments. A tree has two children
        trees 'left_child' and 'right_child' initialized at None.

        Parameters
        ----------
        node: tuple of float
            Node argument contains information on best split to perform at that
            stage: (best_feature, best_metric, best_value).
            best_metric is the value of the split metric for splitting on
            best_feture at best_value.
        probas:
            Probas argument is the empirical class probabilities at that node.
        """
        self.node = node
        self.probas = probas
        self.left_child = None
        self.right_child = None


class Classifier_Tree(object):
    """
    Class defining decision tree classifier learning algorithm.
    """
    def __init__(self, max_depth=5, min_sample_leaf=2, criterion='gini'):
        """
        Initialize classifier max_depth and criterion for computing best binary
        splits.

        Parameters
        ----------
        max_depth: int
            Maximum depth of Decision Tree that will be constructed.
        min_sample_leaf: int
            Minimum number of samples in node required to perform another
            split.
        criterion: str
            Criterion used for computing best splits. Can be either 'gini' or
            'entropy'.
        """
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.criterion = criterion

    def fit(self, x, y, weights=None):
        """
        Fit Decision Tree Classifier of maximum depth 'max_depth' on population
        (x, y).

        Parameters
        ----------
        x: np.array
            Features.
        y: np.array
            Targets.
        weights: np.array
            Weights for samples. If None, all samples have the same weight.
        """
        self.classes = np.unique(y)
        self.bin_tree = Binary_Tree()
        if weights is None:
            weights = np.ones(len(y))
        self.train(x, y, weights, self.max_depth, self.bin_tree)

    def train(self, x, y, weights, depth, bin_tree):
        """
        Construct recursively a Decision Tree on population (x, y).

        Parameters
        ----------
        x: np.array
            Features.
        y: np.array
            Targets.
        weights: np.array
            Weights for samples.
        depth: int
            Tree depth that is left. Also number of iterations left to do in
            recursive funcion.
        bin_tree: Binary_Tree object
            Binary Tree to start from. May be the child of previously constructed
            trees.
        """
        bin_tree.probas = estimate_probas(y, weights, self.classes)

        # Stop fitting when population is size 1, depth argument is O,
        # or remaining population contains only 1 target.
        if (len(x) < self.min_sample_leaf) | (depth == 0) | (len(set(y)) == 1):
            return

        else:
            bin_tree.node = compute_best_feature(x, y, weights, self.criterion)
            best_feature, best_metric, best_value = bin_tree.node

            # Split indexes
            left_idx, right_idx = split(x[:, best_feature], best_value)

            # Construct tree for left population
            left_child = Binary_Tree()
            self.train(x[left_idx], y[left_idx], weights[left_idx],
                       depth-1, left_child)
            bin_tree.left_child = left_child

            # Construct tree for right population
            right_child = Binary_Tree()
            self.train(x[right_idx], y[right_idx], weights[right_idx],
                       depth-1, right_child)
            bin_tree.right_child = right_child

    def predict_probas(self, x_test, y_test):
        """
        Return scores for population (x_test, y_test). For a given sample
        (x, y), its score is the emprical probability of classes the terminal
        leaf it falls in.

        Parameters
        ----------
        x_test: np.array
            Features.
        y_test: np.array
            Targets.
        """
        probas = []
        for x, y in zip(x_test, y_test):
            bin_tree = self.bin_tree
            for i in range(self.max_depth):
                if bin_tree.node is None:
                    break
                feature, metric, value = bin_tree.node
                if x[feature] <= value:
                    bin_tree = bin_tree.left_child
                else:
                    bin_tree = bin_tree.right_child
            probas.append(bin_tree.probas)
        return np.array(probas)

    def predict(self, x_test, y_test):
        """
        Return class predictions for population (x_test, y_test). For a given
        sample (x, y), its prediction is the argmax of the score.

        Parameters
        ----------
        x_test: np.array
            Features.
        y_test: np.array
            Targets.
        """
        probas = self.predict_probas(x_test, y_test)
        return np.array([self.classes[pred] for pred in np.argmax(probas, axis=1)])

    def prune(self):
        """
        Prune tree to reduce overfitting.
        """
        # To be implemented.
        return
