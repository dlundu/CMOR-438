# Logistic Regression

## Overview

Logistic regression is a supervised learning algorithm used for binary classification problems. Instead of predicting a continuous value, it estimates the probability that a data point belongs to a particular class.
The model computes a linear combination of the input features and applies the sigmoid function to map the output to a probability between 0 and 1.

## Key Idea

- Linear model: z = wx + b  
- Activation: sigmoid(z)  
- Output: probability of class 1  

A threshold (typically 0.5) is used to convert probabilities into class predictions.

## Loss Function

The model is trained using **binary cross-entropy loss**, which penalizes incorrect predictions more heavily when the model is confident.

## Application

In this notebook, logistic regression is applied to a classification dataset to predict binary outcomes based on selected features.

## What This Notebook Covers

- Data loading and preprocessing
- Model training
- Visualization of decision boundary
- Performance evaluation (accuracy, confusion matrix)

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
