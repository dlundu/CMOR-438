## K-Nearest Neighbors (KNN)

This project implements the **K-Nearest Neighbors** algorithm to classify medical data using the **Breast Cancer Wisconsin (Diagnostic)** dataset. The final model achieved a 92% classification accuracy.


---

## Implementation 
1. **Distance Metric:** Implemented **Euclidean Distance** to measure proximity between multi-dimensional feature vectors.
2. **Algorithm:** Created a custom prediction function that calculates distances to all training points and selects the majority class among the $K$ closest neighbors.
3. **Hyperparameter Tuning:** Conducted an error-rate analysis across multiple $K$ values (1–21) to find the optimal balance between stability and accuracy (identified K=5). 
4. **Similarity Search:** Extended the logic to act as a "Recommender System," identifying historical cases most similar to a new patient query.

---

## Observations
* **The Scaling Mandate:** Without `StandardScaler`, features like "Mean Area" (values ~1000) would disproportionately influence the distance calculation compared to "Mean Smoothness" (values ~0.1).
* **Stability vs. Accuracy:** We observed that while $K=1$ can be highly accurate on training data, larger odd values of $K$ =5 provide more stable decision boundaries and better generalization to unseen data.
* **Non-Parametric Nature:** Unlike Logistic Regression, this model stores all training data and makes local decisions, allowing it to adapt to complex, non-linear boundaries.

---

##  Dependencies
* `numpy`, `pandas`
* `matplotlib`, `seaborn`
* `scikit-learn` (for data loading and scaling only)
