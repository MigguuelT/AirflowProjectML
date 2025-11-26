import os
import pickle
import shutil
import sys
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.decorators import task

AIRFLOW_HOME = '/Users/migueltorikachvili/PycharmProjects/Airflow_ML'

if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from pipeline.preprocess import preprocess
from pipeline.training import  make_test_train,train_model
from pipeline.evaluation import evaluate_accuracy, save_metrics

# --- VARIÁVEIS DE AMBIENTE E CAMINHOS ---
DATA_DIR = os.path.join(AIRFLOW_HOME, 'data')
ML_DIR = os.path.join(DATA_DIR, 'projeto-ml')
TRAIN_DIR = os.path.join(ML_DIR, 'train')
TEST_DIR = os.path.join(ML_DIR, 'test')
MODEL_DIR = os.path.join(ML_DIR, 'model')
PROCESSED_DIR = os.path.join(ML_DIR, 'processed')
METRICS_DIR = os.path.join(ML_DIR, 'metrics')

with DAG(
    dag_id="training_dag",
    start_date=datetime(2020, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    @task
    def ensure_local_data_exists():
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(TRAIN_DIR, exist_ok=True)
        os.makedirs(TEST_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        print("Diretórios de trabalho locais garantidos.")


    @task
    def load_training_data():
        file_name = 'train.csv'
        file_path = os.path.join(TRAIN_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"AVISO: Arquivo não encontrado - {file_path}.")
            return None
        else:
            print(f"Arquivo CSV encontrado: {file_path}")
            return file_path


    @task
    def preprocess_data():
        train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train.csv'))
        preprocess(train_df).to_csv(os.path.join(PROCESSED_DIR, 'train_processed.csv'), index=False)


    @task
    def split_data():
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_processed.csv'))
        x_train, x_test, y_train, y_test = make_test_train(train_df)

        train_df = pd.DataFrame(x_train)
        train_df['LeaveOrNot'] = y_train
        train_df.to_csv(os.path.join(TRAIN_DIR, 'train_split.csv'), index=False)

        test_df = pd.DataFrame(x_test)
        test_df['LeaveOrNot'] = y_test
        test_df.to_csv(os.path.join(TEST_DIR, 'test.csv'), index=False)


    @task
    def train():
        model_name = 'LogisticRegression'
        df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_split.csv'))
        x_train = df.drop(['LeaveOrNot'], axis=1)
        y_train = df['LeaveOrNot']
        model = train_model(x_train, y_train)
        pickle.dump(model, open(os.path.join(MODEL_DIR, f'model_{model_name}.pkl'), 'wb'))
        return model_name


    @task
    def evaluate_and_log(model_name):
        # 1. Carregar o Modelo usando o nome recebido
        model_file_name = f'model_{model_name}.pkl'
        model_path = os.path.join(MODEL_DIR, model_file_name)

        print(f"Carregando o modelo: {model_path}")

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # 2. Carregar os Dados de Teste
        test_df = pd.read_csv(os.path.join(TEST_DIR, 'test.csv'))
        x_test = test_df.drop(['LeaveOrNot'], axis=1)
        y_test = test_df['LeaveOrNot']

        # 3. Avaliar a Acurácia
        accuracy = evaluate_accuracy(x_test, y_test, model)
        print(f"Acurácia do modelo {model_name}: {accuracy:.4f}")

        # 4. Salvar as Métricas
        save_metrics(METRICS_DIR, model_name, accuracy)


    @task(trigger_rule='all_done')
    def clean_up():
        # remover arquivo de treino
        train_file = os.path.join(TRAIN_DIR, 'train.csv')
        # logica para limpar dados de treino


    # FLUXO DA DAG
    # 1. Defina a saída da task 'train()'
    model_name_output = train()

    # 2. Defina o fluxo da DAG
    (
        ensure_local_data_exists()
        >> load_training_data()
        >> preprocess_data()
        >> split_data()
        >> model_name_output # 'model_name_output' representa a task 'train()'
        >> evaluate_and_log(model_name_output) # Chame a task, passando o resultado
        >> clean_up()
    )
