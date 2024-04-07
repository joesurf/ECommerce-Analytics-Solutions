import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

import mlflow
from mlflow.models import infer_signature


def run_sample_ml_model():
    # df = pd.read_csv('/opt/airflow/data/iris.csv')
    df = pd.read_csv('../../data/iris.csv')

    print("Data Read")

    X = pd.concat([df['sepal_length'], df['sepal_width'], df['petal_length'], df['petal_width']], axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle = True, stratify = y)

    print("Data split")

    params = {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 2
    }



    clf = tree.DecisionTreeClassifier(**params)
    clf = clf.fit(X_train, y_train)

    print("Fit completed")

    y_test_pred = clf.predict(X_test)

    print("Prediction done")

    accuracy = metrics.accuracy_score(y_test, y_test_pred)

    print("Accuracy determined:", accuracy)

    # Set tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://localhost:8081")

    print("Tracking URI set")

    # Create a new MLflow Experiment
    mlflow.set_experiment("BaselineModel")

    print("Experiment name set")

    # Start an MLflow run
    with mlflow.start_run():

        print("Updating mlflow")

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric, in this case we are using accuracy
        mlflow.log_metric("accuracy", accuracy)

        # Set a tag to identify the experiment run
        mlflow.set_tag("Training Info", "Baseline Model - Decision Tree Classifier for Iris Flower Dataset")

        # Infer the model signature
        signature = infer_signature(X_train, clf.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="baseline_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="baseline_model"
        )

        # Note down this model uri to retrieve the model in the future for scoring
        print(model_info.model_uri)


if __name__ == "__main__":
    run_sample_ml_model()