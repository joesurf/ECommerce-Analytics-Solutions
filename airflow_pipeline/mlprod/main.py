import json
import logging

import pandas as pd
import mlflow
from flask import Flask, request, jsonify

LOGGER = logging.getLogger(__name__)



# Create a Flask application instance
app = Flask(__name__)

# Define a route
@app.route('/')
def hello():
    return 'Hello, this is our BT4301 model!'


def load_model(registered_model_name: str):
    mlflow.set_tracking_uri("http://airflow_pipeline-mlflow-server-1:5000")
    models = mlflow.search_registered_models(
        filter_string=f"name = '{registered_model_name}'"
    )
    LOGGER.info(f"Models in the model registry: {models}")
    if models:
        latest_model_version = models[0].latest_versions[0].version
        LOGGER.info(
            f"Latest model version in the model registry used for prediction: {latest_model_version}"
        )
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{registered_model_name}/{latest_model_version}"
        )
        return model
    else:
        LOGGER.warning(
            f"No model in the model registry under the name: {registered_model_name}."
        )


@app.route('/infer', methods=['POST'])
def infer():
    print("Working...")
    input_data = json.loads(request.data)
    input_df = pd.DataFrame(input_data)

    loaded_model = load_model('churn_predictor_logistic_regression')
    
    y_pred = loaded_model.predict(input_df)

    print(y_pred)
    print(type(y_pred))

    return y_pred.tolist()


# Run the application if executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7070, debug=True)
