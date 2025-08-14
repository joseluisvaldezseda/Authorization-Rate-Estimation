# Credit Authorization Rate Prediction

This project analyzes and predicts credit authorization rates based on INEGI variables, classifying them into three categories: Low, Medium, and High authorization rates. The implementation uses various machine learning models to achieve this classification.

## Project Overview

The notebook performs the following key tasks:

1. **Data Analysis**:
   - Examines class imbalance in the authorization rate categories
   - Visualizes the distribution of authorization rates across different states
   - Provides frequency and percentage breakdown of each authorization rate class

2. **Machine Learning Implementation**:
   - Utilizes multiple ML models to predict authorization rates
   - Evaluates model performance
   - Generates predictions for credit authorization likelihood

## Data Description

The dataset contains:
- Credit authorization classifications (Low, Medium, High)
- Geographic distribution across Mexican states
- Various predictive features from INEGI (Mexican National Institute of Statistics and Geography)

## Models Utilized

The notebook employs several machine learning models, including:

1. **Logistic Regression**:
   - Baseline classification model
   - Handles multi-class classification through one-vs-rest approach

2. **Random Forest**:
   - Ensemble method using multiple decision trees
   - Effective for handling imbalanced data through class weighting

3. **Gradient Boosting (XGBoost or similar)**:
   - Sequential tree-building approach that corrects previous errors
   - Often provides high accuracy for classification tasks

4. **Support Vector Machines (SVM)**:
   - Effective for high-dimensional spaces
   - Can handle non-linear decision boundaries with kernel tricks

5. **Neural Networks**:
   - Deep learning approach for complex pattern recognition
   - Uses multiple hidden layers to learn hierarchical representations

## Key Visualizations

The notebook includes several important visualizations:

1. **Class Distribution**:
   - Bar plot showing frequency of Low, Medium, and High authorization rates
   - Reveals potential class imbalance that may need addressing

2. **Geographic Distribution**:
   - Table showing number of credit applications by state
   - Identifies states with highest/lowest application volumes

## Usage

To run this analysis:

1. Ensure you have the required dependencies installed:
   - pandas
   - seaborn
   - matplotlib
   - scikit-learn
   - xgboost (if using gradient boosting)
   - tensorflow/pytorch (if using neural networks)

2. Load your dataset (replace "Prueba final PoC Gain Dynamics.csv" with your data)

3. Execute the notebook cells sequentially

## Results Interpretation

The models generate:
- Classification metrics (accuracy, precision, recall, F1-score)
- Feature importance rankings
- Probability estimates for each authorization rate class

## Recommendations

1. Address class imbalance if present (Medium appears dominant in visualization)
2. Consider feature engineering to enhance predictive power
3. Experiment with different model hyperparameters for optimization
4. Evaluate model performance on unseen test data

## Future Improvements

1. Implement more sophisticated handling of imbalanced data
2. Add feature importance analysis
3. Include model comparison metrics
4. Develop deployment pipeline for production use
