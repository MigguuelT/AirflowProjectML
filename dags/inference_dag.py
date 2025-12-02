import os
import pickle
import sys
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.decorators import task

AIRFLOW_HOME = '/Users/migueltorikachvili/PycharmProjects/Airflow_ML'

if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

# Importa as funções dos módulos
from pipeline.preprocess import preprocess
from pipeline.training import predict_model

# --- VARIÁVEIS DE AMBIENTE E CAMINHOS ---
DATA_DIR = os.path.join(AIRFLOW_HOME, 'data')
ML_DIR = os.path.join(DATA_DIR, 'projeto-ml')
INFERENCE_DIR = os.path.join(ML_DIR, 'inference')
OUTPUT_DIR = os.path.join(ML_DIR, 'output')
DEPLOYMENT_DIR = os.path.join(ML_DIR, 'deployment')


# Task de Setup para garantir que os diretórios de trabalho existam
@task
def ensure_inference_dirs_exists():
    os.makedirs(INFERENCE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Diretórios de trabalho para Inferência garantidos.")


@task
def load_inference_data(**context):  # <--- 1. Mude o argumento para receber o contexto completo
    # O 'ds' é a variável que contém a string da data (YYYY-MM-DD)
    execution_date_str = context['ds']  # <--- 2. Acesse 'ds' dentro do dicionário 'context'

    file_name = f'{execution_date_str}.csv'  # 3. Use a string de data correta
    file_path = os.path.join(INFERENCE_DIR, file_name)

    if not os.path.exists(file_path):
        # A task falhará se não encontrar, mas é importante retornar None para o fluxo
        print(f"AVISO: Arquivo não encontrado - {file_path}. Terminando a execução.")
        return None
    else:
        print(f"Arquivo CSV encontrado: {file_path}")
        return file_path


@task
def load_model():
    model_file_name = 'model.pkl'
    model_path = os.path.join(DEPLOYMENT_DIR, model_file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo de produção não encontrado em {model_path}.")

    return model_path


@task
def preprocess_data(file_path):
    if not file_path:
        return None  # Evita processar se o load_inference_data retornar None

    df = pd.read_csv(file_path)

    # Preprocessamento (sem a coluna target)
    processed_path = os.path.join(INFERENCE_DIR, 'inference_processed.csv')
    preprocess(df, inference=True).to_csv(processed_path, index=False)

    return processed_path


@task
def predict(model_path, processed_path, **context):
    if not processed_path:
        return None  # Evita prever se o preprocess_data retornar None

    df = pd.read_csv(processed_path)
    model = pickle.load(open(model_path, 'rb'))

    # A task predict_model deve receber apenas o DataFrame de features
    df['LeaveOrNot_predicted'] = predict_model(df, model)

    # Use 'ds' para o nome do arquivo de output
    output_date_str = context['ds']

    output_filename = f'employee_predict_{output_date_str}.csv'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_csv(output_path, index=False)

    print(f"✅ Previsões salvas em: {output_path}")

    return processed_path  # Retorna o caminho para a task de limpeza


@task(trigger_rule='all_done')
def finish_job(processed_path=None):
    # O argumento processed_path contém o valor retornado pela task 'predict'.

    if processed_path and os.path.exists(processed_path):
        os.remove(processed_path)  # Comando para deletar o arquivo
        print(f"✅ Arquivo temporário removido: {processed_path}")
    else:
        print("Nenhum arquivo temporário de inferência para remover.")


# --- DEFINIÇÃO DA DAG ---
with DAG(
        dag_id="inference_dag",
        start_date=datetime(2025, 1, 1),
        schedule_interval=timedelta(days=1),  # Sugestão: rodar diariamente
        catchup=False,
        tags=['mlops', 'production'],
) as dag:
    # 1. Obtenção do Caminho do Modelo (Rodada uma única vez)
    model_path = load_model()

    # 2. Fluxo Principal
    inference_file_path = load_inference_data()
    processed_data_path = preprocess_data(inference_file_path)
    prediction_output = predict(model_path, processed_data_path)

    # 3. Definição do Fluxo
    (
            ensure_inference_dirs_exists()
            >> inference_file_path
            >> processed_data_path
            # A task predict depende do modelo e dos dados processados
            >> prediction_output
            # A task de limpeza depende do output da previsão (que retorna o caminho do arquivo temporário)
            >> finish_job(prediction_output)
    )