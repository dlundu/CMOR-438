# DBSCAN
# DBSCAN: Density-Based Clustering for Wholesale Client Segmentation

This project implements **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) to group wholesale clients based on their annual spending across six product categories: Fresh, Milk, Grocery, Frozen, Detergents/Paper, and Delicassen.

### **Theoretical Framework**
DBSCAN identifies clusters as regions of high density separated by regions of low density. Unlike K-Means, it can identify outliers as **noise**.
*   **Core Points**: Points with at least `min_samples` within a distance $\epsilon$.
*   **Border Points**: Points within the neighborhood of a core point but without enough neighbors to be cores themselves.
*   **Noise Points**: Outliers that don't belong to any group (labeled as `-1`).

### **Implementation Details**
*   **Data**: Wholesale Customers dataset (440 clients).
*   **Preprocessing**: `StandardScaler` is used to ensure features with large ranges (like Fresh) don't overwhelm smaller features.
*   **Dimensionality Reduction**: PCA is used to project 6D spending data into 2D for visualization.
*   **Tuning**: Final model uses $\epsilon = 1.0$ and `min_samples = 5`.

### **Dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
