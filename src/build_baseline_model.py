import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os

# --- 0. Setup Report Directory ---
output_dir = '../reports/baseline_model'
os.makedirs(output_dir, exist_ok=True)
print(f"Reports will be saved to: {output_dir}")

# --- 1. Load Raw Data ---
print("\nLoading raw data...")
try:
    df = pd.read_csv("../data/raw/equipment_anomaly_data.csv")
    print(f"Successfully loaded {len(df)} rows from equipment_anomaly_data.csv")
except FileNotFoundError:
    print("Error: '../data/raw/equipment_anomaly_data.csv' not found.")
    exit()

# --- 2. Feature Engineering (Baseline) ---
print("\nPreparing data for baseline modeling...")
features = ['temperature', 'pressure', 'vibration', 'humidity']
target = 'faulty'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.")

# --- 3. Model Training ---
print("\nTraining the baseline Logistic Regression model...")
baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 4. Model Evaluation & Reporting ---
print("\n--- Generating Model Evaluation Report ---")
y_pred = baseline_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Model Accuracy: {accuracy:.4f}")

# Generate and save classification report
report_str = classification_report(y_test, y_pred, target_names=['Normal', 'Faulty'])
report_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Baseline Model Accuracy: {accuracy:.4f}\n\n")
    f.write(report_str)
print(f"Classification report saved to {report_path}")

# Generate and save confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Normal', 'Faulty'], yticklabels=['Normal', 'Faulty'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Baseline Model')
cm_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_path}")

# --- 5. Explainability & Reporting ---
print("\n--- Generating Feature Importance Report ---")
coefficients = baseline_model.coef_[0]
importance_df = pd.DataFrame({'Feature': features, 'Importance': coefficients})
importance_df = importance_df.sort_values(by='Importance', key=abs, ascending=False)

# Save feature importance data to CSV
importance_path_csv = os.path.join(output_dir, 'feature_importance.csv')
importance_df.to_csv(importance_path_csv, index=False)
print(f"Feature importance data saved to {importance_path_csv}")

# Generate and save feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for Baseline Model')
plt.xlabel('Coefficient (Importance)')
plt.ylabel('Feature')
plt.tight_layout()
importance_path_plot = os.path.join(output_dir, 'feature_importance.png')
plt.savefig(importance_path_plot)
plt.close()
print(f"Feature importance plot saved to {importance_path_plot}")
print("\nAll reports for the Baseline Model have been saved.")