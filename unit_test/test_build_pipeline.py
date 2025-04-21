import unittest
import sys
import os
from pathlib import Path

path = Path(os.path.split(__file__)[0])
sys.path.insert(1, str(path.parent) + '/src')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from pipeline.build_pipeline import create_pipeline

class TestCreatePipeline(unittest.TestCase):

    def test_create_pipeline(self):
        # Define input parameters
        num_variables = ['age', 'income']
        cat_variables = ['gender', 'occupation']
        model = LogisticRegression()  # Dummy model for testing
        # Call the function to be tested
        pipe = create_pipeline(num_variables, cat_variables, model)
        # Check if the returned object is a Pipeline
        self.assertIsInstance(pipe, Pipeline)
        # Check the preprocessor step
        self.assertEqual(pipe.named_steps.keys(), {'preprocessor', 'model'})
        # Check the preprocessor step
        preprocessor = pipe.named_steps['preprocessor']
        self.assertIsInstance(preprocessor, ColumnTransformer)
        self.assertEqual(len(preprocessor.transformers), 2)
        # Check the numerical transformer
        numeric_transformer = preprocessor.transformers[0][1]
        self.assertIsInstance(numeric_transformer, Pipeline)
        self.assertEqual(numeric_transformer.steps[0][0], 'scaler')
        # Check the categorical transformer
        categorical_transformer = preprocessor.transformers[1][1]
        self.assertIsInstance(categorical_transformer, Pipeline)
        self.assertEqual(categorical_transformer.steps[0][0], 'onehot')

if __name__ == '__main__':
    unittest.main()
    