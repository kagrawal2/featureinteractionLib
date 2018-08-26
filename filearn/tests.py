"""
Test Suite for feature interaction model training library

- Need to test each function (division, multiplication, one drop, n-drop, powersets)
- Need to different models (sklearn, xgboost, tensorflow, keras, pytorch)
- Need to test regression and classification
- Need to test with noise
- Need to test accepting different hyperparameters for each model.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from filearn import FeatureInteractionLib
import xgboost
import matplotlib

class FeatureInteractionLibTestSuite(object):

    def __init__(self, seed=5):
        self.seed = seed

    def test_division_xgboost_regressor(self):
        print("\nXGBoost Regressor Division")
        n = 100
        x0 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        x1 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        X = np.column_stack([x0, x1])
        Y = x0 / x1

        # Split our dataset to create train and test sets
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=self.seed)

        xgb_regr_model = xgboost.XGBRegressor()

        fiLib = FeatureInteractionLib(X_train, X_test, y_train, y_test, model=xgb_regr_model, validation_metric=LossFunction.mse)
        fiLib.divide_features()

    def test_division_sklearn_linear_regressor(self):
        print("\nLinear Regressor Division")
        n = 100
        x0 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        x1 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        X = np.column_stack([x0, x1])
        Y = x0/x1

        # Split our dataset to create train and test sets
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=self.seed)

        linear_regr_model = LinearRegression()

        fiLib = FeatureInteractionLib(X_train, X_test, y_train, y_test, model=linear_regr_model, validation_metric=LossFunction.mse)
        fiLib.divide_features()


    def test_multiply_pairwise_xgboost_regressor(self):
        print("\nXGBoost Regressor Multiplication")
        n = 100
        x0 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        x1 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        X = np.column_stack([x0, x1])
        Y = x0 * x1

        # Split our dataset to create train and test sets
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=self.seed)

        xgb_regr_model = xgboost.XGBRegressor()

        fiLib = FeatureInteractionLib(X_train, X_test, y_train, y_test, model=xgb_regr_model, validation_metric=LossFunction.mse)
        fiLib.multiply_pairwise_features()

    def test_multiply_pairwise_sklearn_linear_regressor(self):
        print("\nLinear Regressor Pairwise Multiplication")
        n = 100
        x0 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        x1 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        X = np.column_stack([x0, x1])
        Y = x0 * x1

        # Split our dataset to create train and test sets
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=self.seed)

        linear_regr_model = LinearRegression()

        fiLib = FeatureInteractionLib(X_train, X_test, y_train, y_test, model=linear_regr_model, validation_metric=LossFunction.mse)
        fiLib.multiply_pairwise_features()

    def test_multiply_sklearn_linear_regressor(self):
        print("\nLinear Regressor Triple Multiplication")
        n = 100
        x0 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        x1 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        x2 = np.array([1 if x == 0 else x for x in np.random.randint(-100, high=100, size=n)])
        X = np.column_stack([x0, x1, x2])
        Y = x0 * x1 * x2

        # Split our dataset to create train and test sets
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=self.seed)

        linear_regr_model = LinearRegression()

        fiLib = FeatureInteractionLib(X_train, X_test, y_train, y_test, model=linear_regr_model, validation_metric=LossFunction.mse)
        fiLib.multiply_features(3)


class LossFunction(object):

    @staticmethod
    def mse(y_pred, y_test):
        return sum(pow(y_pred - y_test, 2))


# Runs this module file as a script, for testing
if __name__ == "__main__":

    FeatureInteractionLibTestSuite().test_division_xgboost_regressor()
    FeatureInteractionLibTestSuite().test_division_sklearn_linear_regressor()
    FeatureInteractionLibTestSuite().test_multiply_pairwise_xgboost_regressor()
    FeatureInteractionLibTestSuite().test_multiply_pairwise_sklearn_linear_regressor()

    FeatureInteractionLibTestSuite().test_multiply_sklearn_linear_regressor()





