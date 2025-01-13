from sklearn import metrics
import mlflow
from mlflow.models.signature import infer_signature
import time
import os

from module_data_path import mlruns_data_path, plots_data_path

def model_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args,**kwargs)
        t2 = time.time() - t1
        print(f'Model took {t2} seconds')
    return wrapper

@model_time
def model_evaluate(model, X_train, y_train, X_test, y_test):

    mlruns_path = mlruns_data_path()
    mlflow.set_tracking_uri(mlruns_path)
    plots_path = plots_data_path()

    #create a new experiment
    experiment_name = 'LogRegWithMlflow'
    try:
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # start mlflow run
    with mlflow.start_run(experiment_id=exp_id):
        
        # Hyperparameters log
        mlflow.log_param("Model Type", type(model).__name__)
        for hyperparameter, value in model.get_params().items():
            mlflow.log_param(hyperparameter, value)

        # Model training    
        model.fit(X_train, y_train)
        
        # Log evaluation metrics
        # accuracy score
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"{model}: accuracy={accuracy:.2f} %")
        mlflow.log_metric("Accuracy", accuracy)

        # Log artifacts (plots)
        cm = metrics.confusion_matrix(y_test, y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig(os.path.join(plots_path,'confusion_matrix.png'))
        mlflow.log_artifact(os.path.join(plots_path,'confusion_matrix.png'))

        # auc score
        y_probs = model.predict_proba(X_test)[:,1]
        auc_score = metrics.roc_auc_score(y_test, y_probs)
        print(f"{model}: auc={auc_score:.2f} %")
        mlflow.log_metric("AUC score", auc_score)

        # Log the model itself
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        mlflow.end_run()