# Regression Trees

## Overview: Regression Tree Analysis: Ames Housing Dataset

This notebook explores the implementation and evaluation of a Regression Tree using property sales data from Ames, Iowa (kaggle AmesHousing.csv). The project focuses on the relationship between physical housing attributes and final sale prices, emphasizing the bias-variance tradeoff inherent in tree-based models.

The goal of this analysis is to demonstrate how a non-parametric model partitions a complex feature space into distinct regions to predict continuous numerical values.

## Core Algorithm Logic

*   **Recursive Splitting:** The model repeatedly bifurcates the data based on feature thresholds that provide the maximum reduction in Mean Squared Error (MSE).
*   **Leaf Prediction:** Each terminal node (leaf) predicts a constant value derived from the mean of all training observations within that specific region.
*   **Scale Independence:** As a threshold-based learner, this model naturally handles features with different ranges (e.g., *Year Built* vs. *Living Area*) without requiring prior normalization or scaling.

## Technical Implementation

### Dataset
*   **Source:** `AmesHousing.csv`
*   **Target:** `SalePrice` (Continuous)
*   **Key Predictors:** Overall Quality, Ground Living Area, Year Built, and Basement/Floor Square Footage.

### Workflow
1.  **Exploratory Data Analysis:** Visualizing the right-skewed distribution of home prices and identifying high-correlation features via heatmaps.
2.  **Complexity Control:** Using a Validation Curve to analyze how `max_depth` influences the transition from high bias (underfitting) to high variance (overfitting).
3.  **Hyperparameter Tuning:** Utilizing `GridSearchCV` to determine the optimal configuration for tree depth and leaf size.
4.  **Model Diagnostics:** Evaluating performance through Parity Plots (Actual vs. Predicted) and Residual Analysis to identify systematic errors or outliers.

### Key Takeaways
**Primary Drivers:** The tree logic identified Overall Quality and Living Area as the most significant predictors, placing them at the highest levels of the decision.

**Optimization:** A `max_depth` of 10 and `min_samples_leaf` of 10 were found to be the "sweet spot" for balancing predictive power with generalizability.

**Performance Limits:** While highly interpretable, the residual "cloud" for high-value properties highlights the instability of a single tree, suggesting that ensemble methods (like Random Forests) would be the logical next step for increased robustness.

### Dependencies
*   **Languages:** Python 3.x
*   **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`



