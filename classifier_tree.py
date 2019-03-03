from collections import Counter
import numpy as np

from feature_split import compute_best_feature, split


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

    def train(self, x, y, depth, bin_tree):
        """
        Construct recursively a Decision Tree on population (x, y).

        Parameters
        ----------
        x: np.array
            Features.
        y: np.array
            Targets.
        depth: int
            Tree depth that is left. Also number of iterations left to do in
            recursive funcion.
        bin_tree: Binary_Tree object
            Binary Tree to start from. May be the child of previously constructed
            trees.
        """
        count = Counter(y)
        class_count = [count[i] if i in count.keys() else 0.0 for i in self.classes]
        bin_tree.probas = np.array(class_count) / float(sum(class_count))

        # Stop fitting when population is size 1, depth argument is O,
        # or remaining population contains only 1 target.
        if (len(x) < self.min_sample_leaf) | (depth == 0) | (len(count) == 1):
            return

        else:
            bin_tree.node = compute_best_feature(x, y, self.criterion)
            best_feature, best_metric, best_value = bin_tree.node

            # Split indexes
            left_idx, right_idx = split(x[:, best_feature], best_value)

            # Construct tree for left population
            left_child = Binary_Tree()
            self.train(x[left_idx.flatten()], y[left_idx].flatten(),
                       depth-1, left_child)
            bin_tree.left_child = left_child

            # Construct tree for right population
            right_child = Binary_Tree()
            self.train(x[right_idx.flatten()], y[right_idx].flatten(),
                       depth-1, right_child)
            bin_tree.right_child = right_child

    def fit(self, x, y):
        """
        Fit Decision Tree Classifier of maximum depth 'max_depth' on population
        (x, y).

        Parameters
        ----------
        x: np.array
            Features.
        y: np.array
            Targets.
        """
        self.classes = np.unique(y)
        self.bin_tree = Binary_Tree()
        self.train(x, y, self.max_depth, self.bin_tree)

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

    def prune(self):
        """
        Prune tree to reduce overfitting.
        """
        # To be implemented.
        return
