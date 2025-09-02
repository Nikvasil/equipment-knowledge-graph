import time
from rdflib import Graph
import numpy as np
import pandas as pd
import os

# --- 1. Setup and Configuration ---
# Define the output directory and file for the report
output_dir = '../reports/performance'
report_path = os.path.join(output_dir, 'sparql_performance_report.csv')
os.makedirs(output_dir, exist_ok=True)

# Number of times to run each query to get a stable average
NUM_RUNS = 20

# --- 2. Define All Queries to Benchmark ---
# We store queries in a dictionary with friendly names for the final report.
queries_to_benchmark = {
    "Faulty Equipment Readings": """
        PREFIX ont: <http://www.example.com/ontology/equipment#>
        SELECT ?equipmentType ?temp ?pressure ?vibration ?humidity
        WHERE {
          ?reading a ont:Anomaly .
          ?reading ont:isReadingOf ?equipment .
          ?equipment ont:equipmentType ?equipmentType .
          ?reading ont:hasTemperature ?temp .
          ?reading ont:hasPressure ?pressure .
          ?reading ont:hasVibration ?vibration .
          ?reading ont:hasHumidity ?humidity .
        }
    """,
    "Fault Frequency by Location (Aggregate)": """
        PREFIX ont: <http://www.example.com/ontology/equipment#>
        SELECT ?locationName ?equipmentType (COUNT(?anomaly) AS ?faultyCount)
        WHERE {
          ?anomaly a ont:Anomaly .
          ?anomaly ont:isReadingOf ?equipment .
          ?equipment ont:equipmentType ?equipmentType .
          ?equipment ont:isLocatedIn ?location .
          ?location ont:locationName ?locationName .
        }
        GROUP BY ?locationName ?equipmentType
        ORDER BY DESC(?faultyCount)
    """,
    "Correlating Abnormal Vibration": """
        PREFIX ont: <http://www.example.com/ontology/equipment#>
        SELECT ?equipmentType ?vibration ?temp ?pressure
        WHERE {
          ?reading ont:hasVibration ?vibration .
          FILTER(?vibration > 2.0) .
          ?reading ont:isReadingOf ?equipment .
          ?equipment ont:equipmentType ?equipmentType .
          ?reading ont:hasTemperature ?temp .
          ?reading ont:hasPressure ?pressure .
        }
        ORDER BY DESC(?vibration)
    """,
    "Full KG Model Data Extraction": """
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
}

# --- 3. Load the Knowledge Graph ---
print("Loading knowledge graph... (This may take a moment)")
try:
    g = Graph()
    # Ensure this path is correct for your project structure
    g.parse("../data/processed/populated_knowledge_graph.ttl", format="turtle")
    print(f"Graph loaded with {len(g)} triples.")
except FileNotFoundError:
    print("Error: Knowledge graph file not found. Please check the path.")
    exit()

# --- 4. Run the Benchmark ---
all_results = []

print(f"\nStarting benchmark. Each query will be run {NUM_RUNS} times.")

for name, query in queries_to_benchmark.items():
    timings = []
    print(f"\nTesting Query: '{name}'...")

    for i in range(NUM_RUNS):
        start_time = time.time()
        g.query(query)
        end_time = time.time()
        duration = end_time - start_time
        timings.append(duration)

    # Calculate statistics
    avg_time = np.mean(timings)
    std_dev = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)

    # Store results for this query
    all_results.append({
        'Query Name': name,
        'Average Time (s)': avg_time,
        'Std Dev (s)': std_dev,
        'Min Time (s)': min_time,
        'Max Time (s)': max_time,
        'Runs': NUM_RUNS
    })
    print(f"-> Finished. Average time: {avg_time:.4f} seconds")

# --- 5. Record the Results ---
# Convert the results list to a pandas DataFrame
results_df = pd.DataFrame(all_results)

# Save the DataFrame to a CSV file
results_df.to_csv(report_path, index=False)

print("\n--- Benchmark Complete ---")
print("Performance report saved to:", report_path)
print("\nResults Summary:")
print(results_df.to_string())