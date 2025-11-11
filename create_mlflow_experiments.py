import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

experiment_names = ["Churn_Prediction_v1", "Fraud_Detection_v1", "Product_Recommendation_v1", "Sales_Forecasting_v1"]

for name in experiment_names:
    experiment_id = mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"{name}_run"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", random.choice([100, 150, 200]))
        mlflow.log_param("max_depth", random.choice([5, 10, 15]))

        acc = round(random.uniform(0.7, 0.95), 3)
        auc = round(random.uniform(0.75, 0.98), 3)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)

        y_true = [random.randint(0, 1) for _ in range(100)]
        y_pred = [random.randint(0, 1) for _ in range(100)]
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        fig_path = f"{name}_confusion_matrix.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)

        output_path = f"{name}_sample_output.txt"
        with open(output_path, "w") as f:
            f.write(f"Experiment: {name}\nAccuracy: {acc}\nAUC: {auc}\n")
        mlflow.log_artifact(output_path)
        os.remove(output_path)

print("✅ 4 MLflow experiments created with artifacts and metrics.")
