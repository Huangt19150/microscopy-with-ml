import mlflow
import random

samples = ['sample_1', 'sample_2', 'sample_3']
thresholds = [0.3, 0.5, 0.7]

with mlflow.start_run(run_name="test_evaluation_logging"):
    for sample in samples:
        for i, threshold in enumerate(thresholds):
            # Simulate metric values
            f1 = random.uniform(0, 1)
            precision = random.uniform(0, 1)
            recall = random.uniform(0, 1)

            # Use the threshold index as the step to visualize progression
            mlflow.log_metric(f"{sample}_f1", f1, step=i)
            mlflow.log_metric(f"{sample}_precision", precision, step=i)
            mlflow.log_metric(f"{sample}_recall", recall, step=i)


# # Start an MLflow run
# with mlflow.start_run(run_name="test-evaluation-logging-sample-threshold"):

#     for sample in samples:
#         for threshold in thresholds:
#             # Simulate metric values
#             f1 = random.uniform(0, 1)
#             precision = random.uniform(0, 1)
#             recall = random.uniform(0, 1)

#             # Log metrics with a structured name (sampleID/threshold/metric)
#             mlflow.log_metric(f"{sample}_threshold_{threshold}_f1", f1)
#             mlflow.log_metric(f"{sample}_threshold_{threshold}_precision", precision)
#             mlflow.log_metric(f"{sample}_threshold_{threshold}_recall", recall)