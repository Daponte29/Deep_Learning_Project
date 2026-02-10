"""Unit tests for the Deep Learning project"""
import unittest
import torch
from src.models import SimpleCNN
from src.utils import get_device


class TestSimpleCNN(unittest.TestCase):
    """Test cases for SimpleCNN model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 10
        self.model = SimpleCNN(num_classes=self.num_classes)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 28, 28)
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsInstance(self.model, SimpleCNN)
        self.assertEqual(self.model.fc2.out_features, self.num_classes)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        output = self.model(self.input_tensor)
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
    
    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)
    
    def test_model_output_range(self):
        """Test that model outputs reasonable values"""
        output = self.model(self.input_tensor)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_model_training_mode(self):
        """Test model can switch between training and eval modes"""
        self.model.train()
        self.assertTrue(self.model.training)
        
        self.model.eval()
        self.assertFalse(self.model.training)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_get_device(self):
        """Test device detection"""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ['cpu', 'cuda', 'mps'])


if __name__ == '__main__':
    unittest.main()
