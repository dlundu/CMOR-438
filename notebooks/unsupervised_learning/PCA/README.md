# PCA: Principal Component Analysis for Heart Risk Projection

This notebook applies **PCA** to a Heart Disease dataset containing 13 clinical features to reduce dimensionality and visualize patient risk.

### **Analysis of Results**
*   **PC1**: Combines multiple clinical features (age, cholesterol, heart rate) into a single value capturing the highest variance.
*   **PC2**: Captures the second-best combination of features.
*   **Variance Explained**: In this dataset, PC1 captures **21.37%** and PC2 captures **11.97%** of the total variance.

### **Key Takeaways**
PCA successfully creates a 2D coordinate system that preserves the original data structure, though some class-separating information may be lost in the projection since PCA focuses on linear relationships.

### **Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
