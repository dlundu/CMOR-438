# Multilayer Perceptron

### Neural Networks: Hotel Booking Cancellation Prediction 

## **Overview**
This section explores the application of **Multi-Layer Perceptrons (MLP)**, a class of feedforward artificial neural networks, to predict hotel booking cancellations (provided by booking.csv from kaggle). By analyzing behavioral data—such as lead time, pricing, and special requests—this model identifies reservations at high risk of being canceled. This allows hospitality providers to optimize revenue management and room inventory.

---

## **How it Works**
Neural networks function as universal function approximators, structured in three main parts:

*   **Input Layer:** Receives the raw features from the booking dataset.
*   **Hidden Layers:** These layers consist of neurons that apply weights and biases to the data. By passing signals through an activation function (like **ReLU**), the network identifies abstract, non-linear relationships that a standard linear model might overlook.
*   **Output Layer:** Produces a final probability or classification (Canceled vs. Not Canceled).

Learning occurs through **backpropagation** and an optimizer (such as **Adam**). The model calculates the difference between its prediction and the actual outcome, then adjusts the internal weights to minimize error in future iterations.

---

## **Implementation**
In this notebook, I utilized the **`scikit-learn`** framework to build a robust classification pipeline. Key steps included:

*   **Target Encoding:** Converting categorical booking statuses into binary numeric labels.
*   **Feature Engineering:** Removing non-informative identifiers and handling missing values.
*   **Data Normalization:** Applying `StandardScaler` to ensure all numerical features (like price vs. guest count) are on a consistent scale for the network.
*   **Model Architecture:** Implementing an MLP with two hidden layers (64 and 32 neurons) to capture deep dependencies.
*   **Evaluation:** Using Accuracy, Confusion Matrices, and ROC-AUC curves to measure predictive power.

---

## **Performance Results**
The model achieved a final test **accuracy of 83.85%** and an **AUC of 0.891**. While accuracy provides a high-level view, the strong AUC score confirms the model's high sensitivity in distinguishing between guests who intend to stay and those likely to cancel.

---

## **Why use Neural Networks/MLP?**
Traditional models like Logistic Regression assume a linear relationship between features. However, human behavior is often more complex. So for ex., a high room price might not lead to a cancellation on its own, but when combined with a long lead time, the probability changes significantly.

*   **Automatic Feature Interaction:** MLPs "rebalance" and weigh these combinations automatically within their hidden layers.
*   **Non-linearity:** They can model "thresholds" and curves in the data that simpler algorithms miss, making them far more effective for dense, behavioral datasets.

---

## **Dependencies**
To run this notebook, ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
