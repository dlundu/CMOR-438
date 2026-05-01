# Singular Value Decomposition (SVD): Image Compression

This project implements **SVD** to perform image compression by treating a grayscale image as a 2D matrix $A$.

### **Mathematical Framework**
SVD decomposes the matrix into $A = U \Sigma V^T$:
*   **$U$**: Left singular vectors (spatial patterns).
*   **$\Sigma$**: Diagonal matrix of singular values (importance of patterns).
*   **$V^T$**: Right singular vectors (feature variations).

### **Compression Metrics**
We evaluate quality using:
1. **Frobenius Norm Error**: The mathematical "distance" between the original and compressed image.
2. **Cumulative Energy**: The percentage of total variance captured by $k$ components (aiming for ~95% energy).

### **Dependencies**
```bash
pip install numpy matplotlib
