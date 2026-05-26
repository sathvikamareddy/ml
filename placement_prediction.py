# ==========================================
# LOGISTIC REGRESSION - PLACEMENT PREDICTION
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from mlxtend.plotting import plot_decision_regions

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("new_placement_dataset.csv")

# ==========================================
# DISPLAY DATA
# ==========================================

print("\nFIRST 5 RECORDS\n")
print(df.head())

print("\nDATASET INFO\n")
print(df.info())

print("\nSTATISTICAL SUMMARY\n")
print(df.describe())

# ==========================================
# SCATTER PLOT
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df['cgpa'],
    df['iq'],
    c=df['placement'],
    cmap='winter',
    s=100
)

plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("Placement Analysis")

plt.show()

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df[['cgpa', 'iq']]
y = df['placement']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY\n")
print(accuracy)

print("\nCONFUSION MATRIX\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# DECISION BOUNDARY
# ==========================================

plt.figure(figsize=(8,6))

plot_decision_regions(
    X_train,
    y_train.values,
    clf=model,
    legend=2
)

plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("Logistic Regression Decision Boundary")

plt.show()
