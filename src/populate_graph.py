import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

# --- 1. Setup: Define Namespaces and Load Graph ---

# Define namespaces for our ontology (ont) and the data we will create (data)
ONT = Namespace("http://www.example.com/ontology/equipment#")
DATA = Namespace("http://www.example.com/data/equipment/")

# Create a new RDF graph
g = Graph()

# Bind the prefixes to the graph for cleaner output
g.bind("ont", ONT)
g.bind("data", DATA)

# (Optional) Load the base ontology structure from your .owl file
try:
    g.parse("equipment_ontology.rdf", format="xml")
    print("Successfully loaded base ontology from 'equipment_ontology.rdf'.")
except FileNotFoundError:
    print("Base ontology file not found. A new graph will be created.")

# --- 2. Data Loading: Read the CSV File ---

try:
    df = pd.read_csv("../data/raw/equipment_anomaly_data.csv")
    print(f"Successfully loaded {len(df)} rows from CSV.")
except FileNotFoundError:
    print("Error: 'equipment_anomaly_data.csv' not found.")
    exit() # Exit the script if the data file is missing

# --- 3. Transformation: Loop Through CSV and Create Triples ---

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Create unique URIs for each individual for this row
    reading_uri = DATA[f"reading-{index}"]
    equipment_name = str(row['equipment']).replace(' ', '_')
    equipment_uri = DATA[f"equipment-{equipment_name}"]
    location_name = str(row['location']).replace(' ', '_')
    location_uri = DATA[f"location-{location_name}"]

    # Add triples for the SensorReading individual
    g.add((reading_uri, RDF.type, ONT.SensorReading))
    g.add((reading_uri, ONT.hasTemperature, Literal(row['temperature'], datatype=XSD.float)))
    g.add((reading_uri, ONT.hasPressure, Literal(row['pressure'], datatype=XSD.float)))
    g.add((reading_uri, ONT.hasVibration, Literal(row['vibration'], datatype=XSD.float)))
    g.add((reading_uri, ONT.hasHumidity, Literal(row['humidity'], datatype=XSD.float)))

    # Conditionally add the Anomaly type
    if row['faulty'] == 1:
        g.add((reading_uri, RDF.type, ONT.Anomaly))

    # Add triples for the Equipment individual
    g.add((equipment_uri, RDF.type, ONT.Equipment))
    g.add((equipment_uri, ONT.equipmentType, Literal(row['equipment'])))

    # Add triples for the Location individual
    g.add((location_uri, RDF.type, ONT.Location))
    g.add((location_uri, ONT.locationName, Literal(row['location'])))

    # Add triples to link the individuals together
    g.add((reading_uri, ONT.isReadingOf, equipment_uri))
    g.add((equipment_uri, ONT.isLocatedIn, location_uri))

# --- 4. Serialization: Save the Populated Graph ---

output_file = "../data/processed/populated_knowledge_graph.ttl"
g.serialize(destination=output_file, format="turtle")

print(f"\n✅ Knowledge graph population complete!")
print(f"   Saved to '{output_file}'")
print(f"   The graph contains {len(g)} triples.")