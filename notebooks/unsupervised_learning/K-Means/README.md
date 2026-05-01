# K-Means Clustering for Customer Segmentation

This project uses **K-Means Clustering** to identify distinct customer segments within a mall's customer base to help inform targeted marketing strategies.

### **Methodology**
*   **Feature Selection**: Focuses on **Annual Income** and **Spending Score**.
*   **Scaling**: `StandardScaler` ensures distance calculations are not biased by the different raw units of income vs. score.
*   **The Elbow Method**: Determined the optimal number of clusters ($k=5$) by plotting the **Within-Cluster Sum of Squares (WCSS)**.
*   **Evaluation**: Used the **Silhouette Score** to confirm cluster cohesion and separation.

### **Customer Segments Identified**
1. **Standard**: Average income, average spending.
2. **Target**: High income, high spending (Ideal for high-end promotions).
3. **Careful**: High income, low spending.
4. **Sensible**: Low income, low spending.
5. **Spendthrift**: Low income, high spending.

### **Dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
