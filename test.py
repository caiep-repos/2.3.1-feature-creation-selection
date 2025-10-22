import unittest
import numpy as np
import pandas as pd
from assignment import min_max_scaler, standard_scaler, one_hot_encode, select_features_by_correlation

class TestFeatureEngineering(unittest.TestCase):
    
    def test_min_max_scaler(self):
        """Test Min-Max scaling functionality"""
        data = [1, 2, 3, 4, 5]
        scaled_data = min_max_scaler(data)
        expected_data = np.array([0., 0.25, 0.5, 0.75, 1.])
        np.testing.assert_array_almost_equal(scaled_data, expected_data)
        
        # Test with different range
        data2 = [10, 20, 30]
        scaled_data2 = min_max_scaler(data2)
        expected_data2 = np.array([0., 0.5, 1.])
        np.testing.assert_array_almost_equal(scaled_data2, expected_data2)
    
    def test_standard_scaler(self):
        """Test Standard scaling (Z-score normalization) functionality"""
        data = [1, 2, 3, 4, 5]
        scaled_data = standard_scaler(data)
        
        # Check that mean is approximately 0 and std is approximately 1
        self.assertAlmostEqual(np.mean(scaled_data), 0, places=10)
        self.assertAlmostEqual(np.std(scaled_data), 1, places=10)
        
        # Test with constant values - should return NaN
        data2 = [0, 0, 0, 0, 0]  # All zeros
        scaled_data2 = standard_scaler(data2)
        self.assertTrue(np.all(np.isnan(scaled_data2)))  # Should be NaN for constant values
    
    def test_one_hot_encode(self):
        """Test One-Hot encoding functionality"""
        data = ['A', 'B', 'A', 'C']
        encoded = one_hot_encode(data)
        
        # Should be 4x3 matrix (4 samples, 3 categories)
        self.assertEqual(encoded.shape, (4, 3))
        
        # Check specific encoding
        expected = np.array([[1, 0, 0],  # A
                            [0, 1, 0],   # B
                            [1, 0, 0],   # A
                            [0, 0, 1]])  # C
        np.testing.assert_array_equal(encoded, expected)
        
        # Test with predefined categories
        categories = ['A', 'B', 'C', 'D']
        encoded2 = one_hot_encode(data, categories)
        self.assertEqual(encoded2.shape, (4, 4))  # Should include D category
    
    def test_select_features_by_correlation(self):
        """Test feature selection by correlation"""
        # Create sample data
        np.random.seed(42)
        data = np.random.randn(100, 5)  # 100 samples, 5 features
        target = data[:, 0] + 0.5 * data[:, 1] + np.random.randn(100) * 0.1  # Target depends on first 2 features
        
        selected_features = select_features_by_correlation(data, target, threshold=0.1)
        
        # Should select features with correlation > 0.1
        self.assertIsInstance(selected_features, list)
        self.assertGreater(len(selected_features), 0)
        
        # Test with higher threshold
        selected_features_high = select_features_by_correlation(data, target, threshold=0.5)
        self.assertLessEqual(len(selected_features_high), len(selected_features))

if __name__ == '__main__':
    unittest.main()
