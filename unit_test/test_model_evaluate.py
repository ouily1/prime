import unittest
import sys
import os
from pathlib import Path

path = Path(os.path.split(__file__)[0])
sys.path.insert(1, str(path.parent) + '/src')

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from model.model_evaluate import evaluate_model_freq, evaluate_model_cost


class TestModelEvaluate(unittest.TestCase):

    def setUp(self):
        # Create a dummy pipeline for testing
        pipeline = Pipeline(
            [
                (
                    "dummy",
                    DummyClassifier(strategy="stratified", random_state=0),
                )  # Dummy classifier for testing
            ]
        )

        # Generate dummy data for testing
        self.x_test = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )
        self.y_test = pd.Series([0, 1, 0, 1, 0])

        self.pipe = pipeline.fit(self.x_test, self.y_test)
        print(type(self.pipe))

    def test_evaluate_model_freq(self):
        # Call the function to be tested
        matrix, report = evaluate_model_freq(self.pipe, self.x_test, self.y_test)

        # Check the types of returned objects
        self.assertIsInstance(matrix, np.ndarray)
        self.assertIsInstance(report, dict)

        # Check the shape of returned objects
        self.assertEqual(matrix.shape, (2, 2))

        # Check the content of classification report
        expected_report = classification_report(
            self.y_test, self.pipe.predict(self.x_test), 
            output_dict=True
        )
        self.assertEqual(report, expected_report)

    def test_evaluate_model_cost(self):
        # Call the function to be tested
        mse = evaluate_model_cost(self.pipe, self.x_test, self.y_test)

        # Check the type and value of the returned MSE
        self.assertIsInstance(mse, float)
        expected_mse = mean_squared_error(self.y_test, self.pipe.predict(self.x_test))
        self.assertAlmostEqual(mse, expected_mse)


if __name__ == "__main__":
    unittest.main()
