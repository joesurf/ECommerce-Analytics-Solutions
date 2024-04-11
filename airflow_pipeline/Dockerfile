FROM apache/airflow:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /usr/local/airflow/dags

# Set the default command to execute
CMD ["airflow", "webserver"]
