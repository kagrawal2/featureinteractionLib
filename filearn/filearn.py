"""
This module tests a list of features, 
with an optional model for feature interaction,
effects on y

- Check division between features
- Check multiplication between features
- polynomial feature interactions (use sklearn module)
- Check one-drop feature
- Check n-drop features
- Check power sets between features

- Add workflow example: Add a new feature -> run FeatureLib to see improvement...
- Add Pandas support
- Take single feature index or feature column name to test feature interactions on.
 
by Kireet Agrawal, 2018
"""
import numpy as np
import itertools
from operator import mul
from functools import reduce

class FeatureInteractionLib(object):

    def __init__(self, X_train, X_test, y_train, y_test, model, validation_metric):
        # Check if train and test [X, y] lengths are equal
        if len(X_train) != len(y_train):
            raise Exception("Train set mismatch")
        if len(X_test) != len(y_test):
            raise Exception("Test set mismatch")

        # Check if all arrays are type numpy array
        if type(X_train) != np.ndarray or type(X_test) != np.ndarray \
        or type(y_train) != np.ndarray or type(y_test) != np.ndarray:
            raise Exception("Set not numpy.ndarray type")

        # Check if there are features to test (more than 1 features)
        num_samples, num_feats = X_train.shape
        if num_feats < 2:
            raise Exception("Cannot test feaure divide with a single feature.")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.validation_metric = validation_metric

    def divide_features(self):
        """
        Test feature division interaction [(f0 / f1),... (fn-1 / fn)] to the training set, fit model and check on test set
        Compare without division vs. 1. only divided features and 2. appended to original set of features.
        """
        num_samples, num_feats = self.X_train.shape
        if num_feats < 2:
            raise Exception("Cannot test feature divide with a single feature.")

        # No feature division (baseline)
        self.model.fit(self.X_train, self.y_train)
        baseline_test_loss = self.evaluate_test_set(self.X_test, self.y_test)

        # 1. Only feature divisions (baseline)
        division_index_permutations = list(itertools.permutations(range(num_feats), r = 2))
        division_train_set = np.matrix([self.X_train[:, i] / self.X_train[:, j] for i, j in division_index_permutations]).T
        self.model.fit(division_train_set, self.y_train)

        division_test_set = np.matrix([self.X_test[:, i] / self.X_test[:, j] for i, j in division_index_permutations]).T
        division_test_loss = self.evaluate_test_set(division_test_set, self.y_test)

        # 2. Add feature divisions to original feature set
        self.model.fit(np.column_stack([self.X_train , division_train_set]), self.y_train)
        combined_test_loss = self.evaluate_test_set(np.column_stack([self.X_test , division_test_set]), self.y_test)

        print("Baseline Feature Test Loss: {}".format(baseline_test_loss))
        print("Multiplied Features Test Loss: {}".format(division_test_loss))
        print("Combined Features Test Loss: {}".format(division_test_loss))

    def multiply_features(self, k):
        """
        Test feature multiplication interaction [ (f0*f1), (f0*f2),... (f0*f1*...*fn) ] to the training set, fit model and check on test set
        Compare without multiplication vs. 1. only multiplied features and 2. appended to original set of features.
        """
        num_samples, num_feats = self.X_train.shape
        if num_feats < 2:
            raise Exception("Cannot test feature multiply with a single feature.")
        if k > num_feats:
            raise Exception("Cannot multiply more features than the max {}".format(num_feats))

        # No feature multiply (baseline)
        self.model.fit(self.X_train, self.y_train)
        baseline_test_loss = self.evaluate_test_set(self.X_test, self.y_test)

        # 1. Only feature divisions (baseline)
        multiply_index_permutations = list(itertools.permutations(range(num_feats), r = k))

        multiply_train_set = np.matrix([
                reduce(mul, [self.X_train[:, i] for i in inds])
                for inds in multiply_index_permutations]).T
        self.model.fit(multiply_train_set, self.y_train)

        multiply_test_set = np.matrix([
                reduce(mul, [self.X_test[:, i] for i in inds])
                for inds in multiply_index_permutations]).T
        multiply_test_loss = self.evaluate_test_set(multiply_test_set, self.y_test)

        # 2. Add feature divisions to original feature set
        self.model.fit(np.column_stack([self.X_train , multiply_train_set]), self.y_train)
        combined_test_loss = self.evaluate_test_set(np.column_stack([self.X_test , multiply_test_set]), self.y_test)

        print("Baseline Feature Test Loss: {}".format(baseline_test_loss))
        print("Multiplied Features Test Loss: {}".format(multiply_test_loss))
        print("Combined Features Test Loss: {}".format(combined_test_loss))


    def multiply_pairwise_features(self):
        self.multiply_features(k=2)


    def evaluate_test_set(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return self.validation_metric(y_pred, y_test)

        



