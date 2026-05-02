import unittest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class TestDecisionTreeClassifier(unittest.TestCase):

    def setUp(self):
        """Prepare data once for all tests."""
        self.X, self.y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.clf = DecisionTreeClassifier(random_state=42)

    def test_fit_predict_shape(self):
        """Ensure output dimensions match input."""
        self.clf.fit(self.X, self.y)
        predictions = self.clf.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)

    def test_max_depth_constraint(self):
        """Ensure the tree respects the depth limit."""
        depth_limit = 2
        clf = DecisionTreeClassifier(max_depth=depth_limit, random_state=42)
        clf.fit(self.X, self.y)
        # In a binary tree, leaves <= 2^depth
        self.assertLessEqual(clf.get_n_leaves(), 2**depth_limit)

    def test_empty_data_raises_error(self):
        """Ensure proper error handling for empty inputs."""
        with self.assertRaises(ValueError):
            self.clf.fit(np.array([]).reshape(0, 0), [])

    def test_feature_mismatch(self):
        """Ensure error when prediction features don't match training features."""
        self.clf.fit(self.X, self.y)
        with self.assertRaises(ValueError):
            # X has 20 features, we pass 19
            self.clf.predict(self.X[:, :19])

if __name__ == '__main__':
    unittest.main()
