# Logistic Regression

## **Logistic Regression: Smoker Prediction Neuron**

This project implements a **Single Artificial Neuron** from scratch to perform binary classification on a medical insurance dataset. The primary objective is to predict whether a patient is a smoker based on their medical charges and BMI.

---

**Logistic Regression** is a supervised learning algorithm used for **classification**. Unlike linear regression, which predicts a continuous number, logistic regression predicts the **probability** that an input belongs to a specific category, such as Smoker or Non-Smoker. It uses the **Sigmoid function** to "squash" any real-valued number into a range between 0 and 1.

---

### **SectionOverview**
*   **Target Variable**: `smoker` status (0: Non-smoker, 1: Smoker).
*   **Model**: A Single Neuron utilizing the **Sigmoid Activation Function** and **Binary Cross-Entropy Loss**.
*   **Optimization**: Stochastic Graduate Descent (SGD) was used to iteratively update weights and bias.

---

### **Key Steps**
1.  **Exploratory Data Analysis**: Visualized the distribution of medical charges, noting a manual threshold around **$30,000** for separating classes.
2.  **Data Preprocessing**: Applied `StandardScaler` to normalize the `charges` and `bmi` features. This was a critical step to prevent gradient saturation and ensure the model could learn effectively.
3.  **Training**: Trained the neuron using a learning rate ($\alpha$) of **0.01** over **500 epochs**.
4.  **Evaluation**: Visualized the cost function decay and the final sigmoid decision boundary.

---

### **Results**
The initial attempt using raw (unscaled) data resulted in a high error rate of approximately **79.5%**. By implementing feature scaling and optimizing the bias updates (using `float(error)` to avoid NumPy deprecation warnings), the model performance improved significantly, reaching a final **Classification Error of ~9.9%**. This means the model correctly identifies smoking status for roughly **90%** of the patients in the dataset.

---

### **Dependencies**
To run this project, you will need the following Python libraries:
*   **NumPy**: For matrix operations and mathematical functions.
*   **Pandas**: For data manipulation and CSV handling.
*   **Matplotlib / Seaborn**: For generating data visualizations and model plots.
*   **Scikit-learn**: Specifically the `StandardScaler` module for data normalization.
