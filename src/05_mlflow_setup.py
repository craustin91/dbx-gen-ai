import mlflow

mlflow.set_tracking_uri("databricks")

experiment = mlflow.set_experiment(
    "/Shared/genai-rag-lab"
)

print("Experiment ID:", experiment.experiment_id)
print("Experiment name:", experiment.name)