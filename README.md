# Student Placement Prediction using Logistic Regression

## Overview

This project predicts whether a student will get placed based on their CGPA and IQ using the Logistic Regression Machine Learning algorithm.

The project demonstrates the complete Machine Learning workflow including data visualization, preprocessing, model training, prediction, evaluation, and decision boundary visualization.

---

## Features

- Exploratory Data Analysis (EDA)
- Data Visualization using Matplotlib
- Logistic Regression Classification
- Feature Scaling using StandardScaler
- Accuracy Evaluation
- Confusion Matrix
- Decision Boundary Visualization

---

## Dataset Information

| Feature | Description |
|----------|-------------|
| cgpa | Student academic performance |
| iq | Student IQ or aptitude score |
| placement | Placement status (1 = Placed, 0 = Not Placed) |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Mlxtend

---

## Project Workflow

### Data Collection
The dataset is loaded using Pandas.

### Data Visualization
Scatter plots are used to analyze placement distribution.

### Data Preprocessing
- Feature and target selection
- Train-test split
- Feature scaling using StandardScaler

### Model Training
The Logistic Regression model is trained using the training data.

### Model Evaluation
The model performance is evaluated using:
- Accuracy Score
- Confusion Matrix

### Decision Boundary Visualization
Decision regions are plotted to visualize classification boundaries.

---

## Project Structure


placement_prediction.py
new_placement_dataset.csv
README.md
---

