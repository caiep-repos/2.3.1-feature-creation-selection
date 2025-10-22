import numpy as np
import pandas as pd

def min_max_scaler(data):
    """
    Scales a list of numbers to the [0, 1] range.
    
    Args:
        data: List or array of numerical values
        
    Returns:
        numpy array: Scaled values between 0 and 1
    """
    # Your code here
    pass

def standard_scaler(data):
    """
    Scales a list of numbers to have mean=0 and std=1 (Z-score normalization).
    
    Args:
        data: List or array of numerical values
        
    Returns:
        numpy array: Standardized values with mean=0, std=1
    """
    # Your code here
    pass

def one_hot_encode(data, categories=None):
    """
    Converts categorical data to one-hot encoded format.
    
    Args:
        data: List or array of categorical values
        categories: Optional list of all possible categories
        
    Returns:
        numpy array: One-hot encoded matrix
    """
    # Your code here
    pass

def select_features_by_correlation(data, target, threshold=0.1):
    """
    Selects features based on correlation with target variable.
    
    Args:
        data: 2D array or DataFrame of features
        target: 1D array of target values
        threshold: Minimum correlation threshold (default: 0.1)
        
    Returns:
        list: Indices of selected features
    """
    # Your code here
    pass
