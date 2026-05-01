# Decision Trees

## Classification & Regression 

This project explores the implementation of decision trees for both categorical and continuous data. Using Python's `scikit-learn` library, I demonstrate how to visualize model logic, interpret decision boundaries, and optimize performance through depth analysis and the Bias-Variance tradeoff.

### Application
My implementation covers the full machine learning pipeline across two distinct problem types:

### Classification
*   **Datasets:** `make_moons` (artificial binary classification) and the classic **Iris dataset**.
*   **Methodology:** Utilized the `DecisionTreeClassifier` to partition non-linear data into distinct classes.
*   **Visualization:** Implemented `plot_decision_regions` and `plot_tree` to illustrate how the model uses root, decision, and leaf nodes to "carve" the feature space into logical boundaries.
*   **Evaluation:** Performance was measured using **Confusion Matrices** and **Classification Reports** to verify accuracy across various test splits.

### Regression (Continuous Values)
*   **Dataset:** [California Housing Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
*   **Methodology:** Shifted to the `DecisionTreeRegressor` to predict house prices. 
*   **Splitting Criterion:** Unlike classification (which often uses Entropy), this model splits data to minimize **Mean Squared Error (MSE)**.
*   **Goal:** To predict continuous numerical values by ensuring target values within each branch are as homogeneous as possible.

---

## Overfitting vs. Underfitting Analysis
A core component of this project was analyzing the **Bias-Variance Tradeoff** to determine the most effective tree depth for generalizability.

*   **Underfitting:** Testing at `max_depth=1` demonstrated that a shallow tree is too simple to capture the underlying patterns of the housing data (High Bias).
*   **Overfitting:** Testing at `max_depth=20` showed how the tree begins to "memorize" noise, causing training error to plummet while testing error rises (High Variance).
*   **Optimal Complexity:** By iterating through 25 different depths, I identified that **depth 9 or 10** represents the "sweet spot" for the California Housing data—minimizing testing error before the onset of overfitting.

---

## Dependencies
To run this notebook and generate the visualizations, ensure the following libraries are installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
