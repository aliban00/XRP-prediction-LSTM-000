import unittest
import torch
from aipe.core.predictor import Predictor

class TestPredictor(unittest.TestCase):
    def test_architecture_search(self):
        """
        Tests the architecture search method.
        """
        predictor = Predictor()
        model = predictor.search_architecture(None)

        self.assertIsNotNone(model)

        # Test the forward pass
        input_tensor = torch.randn(1, 1)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 1))

    def test_memory_augmentation(self):
        """
        Tests the memory augmentation method.
        """
        predictor = Predictor()
        input_size = 1
        model = predictor.augment_with_memory(input_size)

        self.assertIsNotNone(model)

        # Test the forward pass
        input_tensor = torch.randn(1, 1)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 1))

    def test_uncertainty_quantification(self):
        """
        Tests the uncertainty quantification method.
        """
        predictor = Predictor()

        # Create a simple model with dropout for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(10, 1)
        )

        input_tensor = torch.randn(1, 1)
        mean, std = predictor.quantify_uncertainty(model, input_tensor)

        self.assertIsNotNone(mean)
        self.assertIsNotNone(std)
        self.assertEqual(mean.shape, (1, 1))
        self.assertEqual(std.shape, (1, 1))

    def test_meta_learning(self):
        """
        Tests the meta-learning method.
        """
        predictor = Predictor()

        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

        # Create some dummy task data
        x = torch.randn(1, 1)
        y = torch.randn(1, 1)
        task_data = (x, y)

        adapted_model = predictor.meta_learn(model, task_data)

        self.assertIsNotNone(adapted_model)

        # Check that the parameters have changed
        self.assertFalse(torch.equal(model.state_dict()['0.weight'], adapted_model.state_dict()['0.weight']))

if __name__ == '__main__':
    unittest.main()
