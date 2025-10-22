# Feature Creation and Selection

## Problem Description

In this assignment, you will implement essential feature engineering techniques commonly used in machine learning. You'll work with four fundamental methods: scaling, encoding, and feature selection.

## Learning Objectives

- Understand different feature scaling techniques (Min-Max and Standard scaling)
- Learn how to handle categorical data with one-hot encoding
- Implement feature selection based on correlation analysis
- Apply these techniques to real-world data preprocessing scenarios

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Assignment Tasks

Open the `assignment.py` file and implement the following four functions:

### 1. Min-Max Scaler (`min_max_scaler`)
- **Purpose**: Scale numerical data to the range [0, 1]
- **Formula**: `(x - min) / (max - min)`
- **Use case**: When you know the data bounds and want to preserve the original distribution

### 2. Standard Scaler (`standard_scaler`)
- **Purpose**: Scale data to have mean=0 and standard deviation=1 (Z-score normalization)
- **Formula**: `(x - mean) / std`
- **Use case**: When data follows a normal distribution or when you want to center the data

### 3. One-Hot Encoder (`one_hot_encode`)
- **Purpose**: Convert categorical data into binary vectors
- **Example**: ['A', 'B', 'A'] → [[1,0], [0,1], [1,0]]
- **Use case**: Converting text categories to numerical format for machine learning

### 4. Feature Selector (`select_features_by_correlation`)
- **Purpose**: Select features based on their correlation with the target variable
- **Method**: Keep features with correlation above a threshold
- **Use case**: Removing irrelevant features to improve model performance

## Implementation Tips

- Use NumPy for mathematical operations
- Handle edge cases (e.g., constant values in standard scaling)
- Ensure your functions work with both lists and NumPy arrays
- For one-hot encoding, consider the order of categories

## Testing Your Solution

Run the test file to verify your implementation:

```bash
python test.py
```

The test suite includes:
- Basic functionality tests for each function
- Edge case testing (e.g., constant values, empty arrays)
- Shape and data type validation
- Correlation-based feature selection validation

## Expected Output

All tests should pass when your implementation is correct. Each function should handle the provided test cases and edge cases appropriately.
