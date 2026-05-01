
# Linear Regression Model

Linear regression is a supervised learning tool used to predict a continuous number. In this project, I used it to predict **medical insurance costs** based on person-to-person data.

The model follows a basic equation:
**y = wx + b**

*   **y**: The cost we want to predict (target).
*   **x**: The info we have, like age or BMI (input features).
*   **w**: The "weight" or coefficient that shows how much each piece of info matters.
*   **b**: The "bias" or intercept (the starting point of the line).

## Implementation
I created a **Single Neuron** to represent this model. While also uaing a library, I also created code to make the neuron learn the best-fit line on its own.

*   **Scaling:** I used `StandardScaler` to normalize the data. This is important because insurance prices are  larger compared to ages; scaling keeps the math from breaking during training.
*   **Training:** I used **Stochastic Gradient Descent (SGD)**. The neuron looks at the data point by point, makes a guess, and fixes its weights slightly to get closer to the real answer.
*   **Fixing the Error:** I tracked the **Mean Squared Error (MSE)**. As the neuron learned over many epochs, the error dropped, showing that the predictions were getting better.

## Dependencies
To run this, you will need these standard Python libraries:
*   `numpy` and `pandas` for handling data and math.
*   `matplotlib` and `seaborn` for making the graphs.
*   `scikit-learn` for scaling the features.

## Goal
The goal of this project was to model linear regression, use a single artificial neuron to solve a real-world medical data problem by finding a linear relationship between patient data and their bills.
