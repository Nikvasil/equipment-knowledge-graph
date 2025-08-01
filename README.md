# **Semantic Knowledge Graphs for Explainable AI in Industrial Process Analytics**

Author: Mykyta Vasyliev  
Date: August 2025  
Project Status: In Progress

## **1\. Project Overview**

This repository contains the proof-of-concept system for the Master Thesis titled, "Semantic Knowledge Graphs for Explainable AI: Comparing Feature Extraction Approaches in Industrial Process Analytics."

The project investigates the integration of semantic knowledge graphs with lightweight AI models to improve predictive accuracy and interpretability in manufacturing decision-making. It aims to demonstrate that by enriching raw industrial data with semantic context, AI models can provide more accurate and, crucially, more explainable insights for tasks like anomaly detection and predictive maintenance.

The core workflow involves:

1. Transforming raw telemetric data from a CSV file into an RDF-based knowledge graph using a custom ontology.  
2. Storing the graph in a native triple store (GraphDB).  
3. Using SPARQL to query the graph and extract semantically rich features based on defined competency questions.  
4. Training and comparing two AI models: one on the raw data and one on the semantically extracted features.

## **2\. Setup and Installation**

To set up the project environment and reproduce the results, follow these steps:

### **Prerequisites**

* Python 3.8+  
* GraphDB (Free Edition) installed and running.

### **Installation**

1. **Clone the repository:**  
   git clone \<your-repository-url\>  
   cd equipment-kg

2. Install Python dependencies:  
   (A requirements.txt file should be created for this. For now, the main dependency is rdflib and pandas).  
   pip install pandas rdflib

3. **Set up GraphDB:**  
   * Start the GraphDB server.  
   * Create a new repository with the ID thesis-project.  
   * Select the **OWL-Horst (Optimized)** ruleset during creation.

## **3\. Workflow: How to Reproduce Results**

Follow these steps in order to generate the knowledge graph and extract the feature sets.

### **Step 1: Populate the Knowledge Graph**

Run the populate\_graph.py script from the root directory of the project. This script will:

1. Read the raw data from data/raw/equipment\_anomaly\_data.csv.  
2. Transform the data into RDF triples according to the schema in ontology/equipment\_ontology.rdf.  
3. Generate the populated knowledge graph and save it to data/processed/populated\_knowledge\_graph.ttl.

python src/populate\_graph.py

### **Step 2: Import Data into GraphDB**

1. Navigate to your GraphDB instance (usually http://localhost:7200).  
2. Go to **Import \> RDF**.  
3. Click **Upload RDF files** and select the newly generated data/processed/populated\_knowledge\_graph.ttl.  
4. Click **Import** to load the data into your thesis-project repository.

### **Step 3: Extract Features with SPARQL**

1. Navigate to the **SPARQL** tab in the GraphDB interface.  
2. Execute the SPARQL queries (found in the sparql/ directory or project documentation).  
3. Use the **Export** function in the SPARQL view to save the results as CSV files in the data/processed/ directory.

### **Step 4: Train AI Models (Future Work)**

The scripts for training the AI models will be located in the src/ directory. They will use the CSV files generated in the previous step as input.

*This README is a living document and will be updated as the project progresses.*