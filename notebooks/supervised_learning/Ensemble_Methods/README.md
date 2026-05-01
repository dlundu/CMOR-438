# Ensemble Methods

### Ensemble Learning: Bagging, Random Forests, and Boosting

This notebook implements ensemble learning methods to improve classification and regression performance. Ensemble methods aggregate the predictions of multiple "weak learners" (typically decision trees) to create a single "strong learner" with lower variance and higher accuracy.

---

## What We Do in This Notebook
1. **Hard Voting:** Combine diverse models (Logistic Regression, RF, SVM) to see how a "majority vote" impacts accuracy.
2. **Bagging:** Train 500 decision stumps in parallel using bootstrap sampling to reduce model variance.
3. **Random Forests:** Introduce feature randomness to the bagging process to create more diverse, uncorrelated trees.
4. **Boosting:** Implement sequential learning where each new model corrects the errors of the previous one.

---

## Implementation & Key Parts

### Model Training & Comparison
Will utilise the **Palmer Penguins** dataset to classify species based on morphological measurements.
* **Feature Selection:** We start with two features (`bill_length_mm`, `flipper_length_mm`) for visualization and expand to four features for performance tuning.
* **Performance:** We observe that while single trees can be "jagged" and overfit, ensemble methods like Random Forest and Bagging provide smoother decision boundaries and high generalization.

### AdaBoost Implementation
* **Process:** train an `AdaBoostClassifier` using decision stumps. The algorithm sequentially adjusts the weights of misclassified samples, forcing the next tree to focus on "hard" cases.
* **Observation:**  evaluate performance using a **Classification Report** and a **Confusion Matrix** to track precision and recall for each species.

### Gradient Boosting (Regression)
* **Process:** simulate a non-linear quadratic relationship and train a sequence of `DecisionTreeRegressors`.
* **Observation:** Each tree is fit to the **residual errors** (actual - predicted) of the previous stage. Summing these trees allows the model to perfectly "recover" the non-linear curve.

---

## Performance Summary
* **Accuracy:** ensembles achieved near-perfect accuracy (**~0.99-1.00**) due to the distinct features of the Adelie and Gentoo species.
* **Feature Importance:** Random Forest identified **Flipper Length** and **Bill Length** as the most critical predictors for classification.

---


## Dependencies
To run this notebook, I used the following libraries:
* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`, `mlxtend`
