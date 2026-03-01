import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("noshowappointments.csv")

print("Columns in dataset:")
print(df.columns)

# -----------------------------
# Convert target column
# -----------------------------
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

# -----------------------------
# Select correct features (based on your dataset)
# -----------------------------
X = df[['Age',
        'SMS_received',
        'Scholarship',
        'Hipertension',
        'Diabetes',
        'Alcoholism',
        'Handcap']]

y = df['No-show']

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Decision Tree (balanced for imbalanced data)
# -----------------------------
model = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.title("Feature Importance for No-Show Prediction")
plt.tight_layout()

# Save as image
plt.savefig("feature_importance.png")

plt.show()