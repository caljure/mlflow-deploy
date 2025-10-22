import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import traceback

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Definir rutas ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Artifact Location: {artifact_location} ---")

# --- Asegurar existencia del directorio ---
os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o establecer experimento ---
experiment_name = "CI-CD-Lab2"
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Creado experimento '{experiment_name}' con ID {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    print(f"--- Debug: Experimento '{experiment_name}' ya existe ---")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"--- Debug: Usando experimento existente ID: {experiment_id} ---")

# --- Entrenamiento del modelo ---
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Run de MLflow ---
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"--- Debug: Run ID: {run_id} ---")

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ Modelo registrado en MLflow con MSE: {mse:.4f}")

except Exception as e:
    print(f"--- ERROR durante ejecución de MLflow ---")
    traceback.print_exc()
    sys.exit(1)

# --- Guardar el modelo localmente ---
joblib.dump(model, "model.pkl")
print("--- ✅ Modelo guardado correctamente como 'model.pkl' ---")
