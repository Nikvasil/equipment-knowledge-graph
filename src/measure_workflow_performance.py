import time
import pandas as pd
import numpy as np
from rdflib import Graph
from joblib import load
import os
import random

# --- 1. Setup and Configuration ---
output_dir = '../reports/performance'
report_path = os.path.join(output_dir, 'workflow_performance_report.csv')
os.makedirs(output_dir, exist_ok=True)

# Number of simulated readings to test
NUM_RUNS = 500

# --- 2. Load All Necessary Assets ---
print("Loading assets (KG, model, scaler, columns)...")
try:
    g = Graph()
    g.parse("../data/processed/populated_knowledge_graph.ttl", format="turtle")

    assets_dir = '../models'
    model = load(os.path.join(assets_dir, 'kg_model.joblib'))
    scaler = load(os.path.join(assets_dir, 'kg_scaler.joblib'))
    training_columns = load(os.path.join(assets_dir, 'training_columns.joblib'))

    print("All assets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading assets: {e}")
    print("Please run the 'build_kg_model.py' script first to generate the necessary .joblib files.")
    exit()


# --- 3. Define the End-to-End Workflow Function ---
def process_new_reading(reading_data, graph, ml_model, data_scaler, train_cols):
    """
    This function encapsulates the entire workflow for a single new data point.
    """
    # a) Get context from Knowledge Graph
    query = f"""
    PREFIX ont: <http://www.example.com/ontology/equipment#>
    PREFIX data: <http://www.example.com/data/equipment/>
    SELECT ?equipmentType ?locationName
    WHERE {{
      data:{reading_data['equipment_id']} ont:equipmentType ?equipmentType ;
                                         ont:isLocatedIn ?location .
      ?location ont:locationName ?locationName .
    }}
    """
    results = graph.query(query)
    context = next(iter(results), None)

    if not context:
        return "Error: Equipment ID not found", 0

    # b) Combine raw data with KG context and create a DataFrame
    full_data = {
        'temperature': reading_data['temperature'],
        'pressure': reading_data['pressure'],
        'vibration': reading_data['vibration'],
        'humidity': reading_data['humidity'],
        'equipmentType': str(context.equipmentType),
        'locationName': str(context.locationName)
    }
    df = pd.DataFrame([full_data])

    # c) Preprocess the data (one-hot encode and scale)
    df_processed = pd.get_dummies(df).reindex(columns=train_cols, fill_value=0)
    df_scaled = data_scaler.transform(df_processed)  # Use the loaded scaler

    # d) Get a prediction
    prediction = ml_model.predict(df_scaled)[0]

    return "Anomaly" if prediction == 1 else "Normal"


# --- 4. Run the Benchmark ---
timings = []
print(f"\nStarting benchmark. Testing {NUM_RUNS} simulated readings...")

for i in range(NUM_RUNS):
    # Simulate a new reading from a random piece of equipment
    # (equipment-101 to equipment-105 are Turbines in your data)
    mock_reading = {
        'temperature': random.uniform(80, 120),
        'pressure': random.uniform(2.5, 3.5),
        'vibration': random.uniform(1.0, 3.0),
        'humidity': random.uniform(30, 50),
        'equipment_id': f'equipment-{random.randint(1, 200)}'
    }

    start_time = time.time()
    process_new_reading(mock_reading, g, model, scaler, training_columns)
    end_time = time.time()

    duration = end_time - start_time
    timings.append(duration)

print("Benchmark finished.")

# --- 5. Calculate Statistics and Record the Results ---
avg_time = np.mean(timings)
std_dev = np.std(timings)
min_time = np.min(timings)
max_time = np.max(timings)
p95 = np.percentile(timings, 95)  # 95% of requests were faster than this
p99 = np.percentile(timings, 99)  # 99% of requests were faster than this

results = {
    'Metric': ['Average Time (s)', 'Std Dev (s)', 'Min Time (s)', 'Max Time (s)', '95th Percentile (s)',
               '99th Percentile (s)', 'Total Runs'],
    'Value': [avg_time, std_dev, min_time, max_time, p95, p99, NUM_RUNS]
}
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv(report_path, index=False)

print("\n--- Benchmark Complete ---")
print("Workflow performance report saved to:", report_path)
print("\nResults Summary:")
print(results_df.to_string(index=False))