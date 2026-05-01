# Perceptron

This project implements a **Perceptron** to perform binary classification on two distinct datasets: the **Iris** dataset and the **Palmer Penguins** dataset. The goal is to demonstrate how a single neuron model identifies a linear decision boundary to separate classes.

---

The **Perceptron** is a fundamental supervised learning algorithm for binary classification. It functions by taking input features, applying weights and a bias, and passing the result through a **sign activation function**. If the stimulus is intense enough, the neuron "fires," returning a value of **1** or **-1**.

---

### **Project Overview**
*   **Datasets**: Used the **Iris dataset** (Setosa vs. Versicolor) and the **Palmer Penguins dataset** (penguins.csv).
*   **Features**: Analyzed physical measurements such as **sepal length/width** and **bill length/depth**.
*   **Model Implementation**: A custom Python class implementation including `train`, `net_input`, and `predict` methods.
*   **Optimization**: Employed error-correction learning over a set number of **epochs** to adjust weights and bias.

---

### **Key Steps**
1.  **Data Preprocessing**: Converted categorical species names into numerical labels (**1** and **-1**) and selected relevant features.
2.  **Training**: Initialized weights randomly and iteratively updated them based on misclassifications.
3.  **Visualization**: Generated **decision region plots** to show the linear separator and **error plots** to track convergence over time.

---

### **Conclusion & Findings**
Through implementing this model, I observed the Perceptron's fundamental reliance on **linear separability**. While the **Iris** model converged to **100% accuracy**, the **Penguin** model reached **87.85%** due to natural feature overlap between species. The resulting **oscillation** in the error plots and the imperfect **decision boundaries** highlight the Perceptron's limitations with non-linearly separable data, suggesting that more complex **models** are required for messy, real-world datasets.

---

### **Dependencies**
*   **NumPy**: For matrix operations.
*   **Pandas**: For data manipulation.
*   **Matplotlib / Seaborn**: For data visualization.
*   **Mlxtend**: For plotting decision regions.
