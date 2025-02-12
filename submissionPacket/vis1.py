from graphviz import Digraph

# Create a directed graph
flowchart = Digraph(format="png", name="ML_Training_Flowchart")

# Define nodes
flowchart.node("TD", "Training Data", shape="box")
flowchart.node("Train", "Train Data (0.9)", shape="box")
flowchart.node("Val", "Val Data (0.1)", shape="box")
flowchart.node("Model", "Model", shape="box")
flowchart.node("Loop", "Training Loop", shape="ellipse")
flowchart.node("HyperOpt", "Hyperparameter Optimizer", shape="box")
flowchart.node("Optuna", "Optuna", shape="box")
flowchart.node("RandSearch", "Random Search", shape="box")
flowchart.node("FModel", "Finalized Model", shape="box")
flowchart.node("TestData", "Test Data", shape="box")
flowchart.node("Output", "O/P", shape="box")
flowchart.node("Evaluator", "Evaluator", shape="box")
flowchart.node("Stats", "Statistics", shape="box")

# Connect nodes
flowchart.edges([
    ("TD", "Train"), ("TD", "Val"),  # Training data splits
    ("Train", "Model"), ("Val", "Model"),  # Data feeds into Model
    ("Model", "Loop"), ("Loop", "Model"),  # Training loop
    ("Optuna", "HyperOpt"), ("RandSearch", "HyperOpt"),  # Hyperparameter optimization
    ("HyperOpt", "Model"),  # Optimizer refines model
    ("Model", "FModel"),  # Finalized model
    ("FModel", "TestData"),  # Test data connects to finalized model
    ("FModel", "Output"),  # Finalized model generates output
    ("Output", "Evaluator"), ("Evaluator", "Stats"),  # Evaluation and statistics
])

# Render the flowchart to a file
output_path = "/mnt/data/ML_Training_Flowchart"
flowchart.render(output_path, cleanup=True)

output_path + ".png"
