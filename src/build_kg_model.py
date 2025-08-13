import pandas as pd
from rdflib import Graph
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os

# --- 0. Setup Report Directory ---
output_dir = '../reports/kg_model'
os.makedirs(output_dir, exist_ok=True)
print(f"Reports will be saved to: {output_dir}")

# --- 1. Load Data from Knowledge Graph ---
print("\nLoading knowledge graph...")
try:
    g = Graph()
    g.parse("../data/processed/populated_knowledge_graph.ttl", format="turtle")
    print(f"Graph loaded with {len(g)} triples.")
except FileNotFoundError:
    print("Error: '../data/processed/populated_knowledge_graph.ttl' not found.")
    exit()

# SPARQL query to get a complete, labeled dataset
sparql_query = """
PREFIX ont: <http://www.example.com/ontology/equipment#>
SELECT ?equipmentType ?locationName ?temperature ?pressure ?vibration ?humidity ?isAnomaly
WHERE {
  ?reading a ont:SensorReading ;
           ont:hasTemperature ?temperature ;
           ont:hasPressure ?pressure ;
           ont:hasVibration ?vibration ;
           ont:hasHumidity ?humidity ;
           ont:isReadingOf ?equipment .
  ?equipment ont:equipmentType ?equipmentType .
  ?equipment ont:isLocatedIn ?location .
  ?location ont:locationName ?locationName .
  BIND(EXISTS {?reading a ont:Anomaly} AS ?isAnomaly_bool)
  BIND(IF(?isAnomaly_bool, 1, 0) AS ?isAnomaly)
}
"""

print("\nQuerying graph for training data...")
results = g.query(sparql_query)
data = [{'equipmentType': str(row.equipmentType), 'locationName': str(row.locationName), 'temperature': float(row.temperature), 'pressure': float(row.pressure), 'vibration': float(row.vibration), 'humidity': float(row.humidity), 'isAnomaly': int(row.isAnomaly)} for row in results]
df = pd.DataFrame(data)
print(f"Successfully created DataFrame with {len(df)} rows.")

# --- 2. Feature Engineering & Preprocessing ---
print("\nPreparing data for modeling...")
df_processed = pd.get_dummies(df, columns=['equipmentType', 'locationName'], drop_first=True)
X = df_processed.drop('isAnomaly', axis=1)
y = df_processed['isAnomaly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.")

# --- 3. Model Training ---
print("\nTraining the Logistic Regression model...")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 4. Model Evaluation & Reporting ---
print("\n--- Generating Model Evaluation Report ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Generate and save classification report
report_str = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])
report_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    f.write(report_str)
print(f"Classification report saved to {report_path}")

# Generate and save confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KG-Aware Model')
cm_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_path}")

# --- 5. Explainability & Reporting ---
print("\n--- Generating Feature Importance Report ---")
feature_names = X.columns
coefficients = model.coef_[0]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
importance_df = importance_df.sort_values(by='Importance', key=abs, ascending=False)

# Save feature importance data to CSV
importance_path_csv = os.path.join(output_dir, 'feature_importance.csv')
importance_df.to_csv(importance_path_csv, index=False)
print(f"Feature importance data saved to {importance_path_csv}")

# Generate and save feature importance plot
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='rocket')
plt.title('Feature Importance for KG-Aware Model')
plt.xlabel('Coefficient (Importance)')
plt.ylabel('Feature')
plt.tight_layout()
importance_path_plot = os.path.join(output_dir, 'feature_importance.png')
plt.savefig(importance_path_plot)
plt.close()
print(f"Feature importance plot saved to {importance_path_plot}")
print("\nAll reports for the KG-Aware Model have been saved.")