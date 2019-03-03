import numpy as np

from classifier_tree import Classifier_Tree


class AdaBoostClassifier(object):
    """
    Class defining AdaBoost Classifier algorithm with decision trees.
    """
    def __init__(self, max_depth=5, min_sample_leaf=2, criterion='gini'):
        """
        Initialize AdaBoost Classifier.

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

    def fit(self, x, y, n_trees):
        """
        Fit AdaBoost Classifier on population (x, y) using 'n_trees' Trees.

        Parameters
        ----------
        x: np.array
            Features.
        y: np.array
            Targets.
        n_trees: int
            Number of trees to fit for final ensembling.
        """
        self.trees = []
        self.alphas = []
        weights = np.ones(len(y)) / len(y)
        self.classes = np.unique(y)
        for i in range(n_trees):
            print('===============================')
            print('Training Tree number: ' + str(i))
            print('-------------------------------')
            tree = Classifier_Tree(self.max_depth,
                                   self.min_sample_leaf,
                                   self.criterion)
            tree.fit(x, y, weights)
            self.trees.append(tree)
            preds = np.argmax(tree.predict_probas(x, y), axis=1)
            y_pred = [self.classes[pred] for pred in preds]
            y_err = (y != y_pred)
            err_i = np.sum(y_err * weights) / np.sum(weights)
            print(err_i)
            alpha_i = np.log((1 - err_i) / err_i)
            self.alphas.append(alpha_i)
            weights = weights * np.exp(alpha_i * y_err)

    # TODO: check that
    def predict_probas(self, x, y):
        """
        Return prediction scores on population (x, y).

        Parameters
        ----------
        x: np.array
            Features.
        y: np.array
            Targets.
        """
        probas = 0
        for tree_i, alpha_i in zip(self.trees, self.alphas):
            probas += tree_i.predict_probas(x, y) * alpha_i
        # Normalize scores for sum == 1
        return probas / np.sum(probas, axis=1).reshape(-1, 1)

    def predict(self, x, y):
        """
        Return class predictions on population (x, y).

        Parameters
        ----------
        x: np.array
            Features.
        y: np.array
            Targets.
        """
        probas = self.predict_probas(x, y)
        return np.array([self.classes[pred] for pred in np.argmax(probas, axis=1)])
