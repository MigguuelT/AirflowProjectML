import os
import pickle
import sys
import pandas as pd
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# --- IMPORTAÇÕES DO PIPELINE ---
AIRFLOW_HOME = '/Users/migueltorikachvili/PycharmProjects/Airflow_ML'
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from pipeline.preprocess import preprocess
from pipeline.evaluation import evaluate_accuracy, evaluate_roc_auc, log_production_metrics

# --- VARIÁVEIS DE AMBIENTE E CAMINHOS ---
DATA_DIR = os.path.join(AIRFLOW_HOME, 'data')
ML_DIR = os.path.join(DATA_DIR, 'projeto-ml')
EVAL_DIR = os.path.join(ML_DIR, 'evaluation')
METRICS_DIR = os.path.join(ML_DIR, 'metrics')
DEPLOYMENT_DIR = os.path.join(ML_DIR, 'deployment')


@task
def ensure_evaluation_dirs_exists():
    os.makedirs(EVAL_DIR, exist_ok=True)
    print("Diretórios de trabalho para Avaliação garantidos.")


@task
def load_model():
    model_file_name = 'model.pkl'
    model_path = os.path.join(DEPLOYMENT_DIR, model_file_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo de produção não encontrado em {model_path}.")
    return model_path


@task
def load_evaluation_data(**context):
    # Acessa a data de execução usando **context
    execution_date_str = context['ds']
    file_name = f'eval_{execution_date_str}.csv'
    file_path = os.path.join(EVAL_DIR, file_name)

    if not os.path.exists(file_path):
        print(f"AVISO: Arquivo de avaliação não encontrado - {file_path}.")
        return None
    else:
        print(f"Arquivo de avaliação CSV encontrado: {file_path}")
        return file_path


@task
def preprocess_evaluation_data(file_path):
    if not file_path:
        return None

    df = pd.read_csv(file_path)
    # inference=False garante que a coluna 'LeaveOrNot' seja mantida para avaliação
    processed_path = os.path.join(EVAL_DIR, 'evaluation_processed.csv')
    preprocess(df, inference=False).to_csv(processed_path, index=False)

    return processed_path


@task
def evaluate_production_model(model_path, processed_path):
    if not processed_path:
        return None

    df = pd.read_csv(processed_path)
    model = pickle.load(open(model_path, 'rb'))

    target_column = 'LeaveOrNot'
    x_eval = df.drop([target_column], axis=1)
    y_eval = df[target_column]

    # Avaliar Métricas e Salvar Log
    accuracy = evaluate_accuracy(x_eval, y_eval, model)
    roc_auc = evaluate_roc_auc(x_eval, y_eval, model)
    log_production_metrics(METRICS_DIR, accuracy, roc_auc)

    print(f"✅ Avaliação de Produção Concluída. Acc: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

    # Retorna o caminho do arquivo temporário para limpeza
    return processed_path


@task
def check_performance_threshold(processed_path_from_eval):
    """
    Verifica se as métricas de produção atendem aos limites mínimos (Acc > 0.6 E ROC AUC > 0.6).
    Retorna o task_id da próxima ação.
    """
    if not processed_path_from_eval:
        return 'finish_evaluation_job'  # Não há dados, finaliza.

    # Relê o último registro do log (que acabou de ser salvo por evaluate_production_model)
    log_path = os.path.join(METRICS_DIR, 'production_evaluation_metrics.csv')
    df_logs = pd.read_csv(log_path)
    latest_metrics = df_logs.iloc[-1]

    accuracy = latest_metrics['accuracy']
    roc_auc = latest_metrics['roc_auc']

    MIN_THRESHOLD = 0.6

    if accuracy > MIN_THRESHOLD and roc_auc > MIN_THRESHOLD:
        print(f"✅ PERFORMANCE OK. Metas atingidas. Mantendo o modelo atual.")
        return 'model_is_good_log'  # Task ID para o caminho OK
    else:
        print(f"🚨 ALERTA DE PERFORMANCE. Queda de métricas. Retreino recomendado.")
        return 'retrain_recommended'  # Task ID para acionar o trigger


@task
def model_is_good_log():  # Task para logar o caminho OK
    print("Métricas satisfatórias. O modelo de produção permanece em uso.")
    # Retorna o ID da task de limpeza
    return "finish_evaluation_job"


@task
def finish_evaluation_job(processed_path=None):
    if processed_path and os.path.exists(processed_path):
        os.remove(processed_path)
        print(f"✅ Arquivo temporário de avaliação removido: {processed_path}")
    else:
        print("Nenhum arquivo temporário de avaliação para remover.")


# --- DEFINIÇÃO DA DAG ---
with DAG(
        dag_id="evaluation_dag",
        start_date=datetime(2025, 1, 1),
        schedule_interval=timedelta(days=1),
        catchup=False,
        tags=['mlops', 'evaluation'],
) as dag:
    # 1. SETUP E CARREGAMENTO
    setup_task = ensure_evaluation_dirs_exists()
    model_path_ref = load_model()

    # 2. FLUXO DE DADOS
    evaluation_file_path = load_evaluation_data()
    processed_eval_path = preprocess_evaluation_data(evaluation_file_path)

    # 3. AVALIAÇÃO E DECISÃO
    # evaluation_output armazena a referência da task E o processed_path
    evaluation_output = evaluate_production_model(model_path_ref, processed_eval_path)
    decision = check_performance_threshold(evaluation_output)

    # 4. AÇÕES CONDICIONAIS
    trigger_retrain = TriggerDagRunOperator(
        task_id="retrain_recommended",  # Task ID referenciado pelo 'decision'
        trigger_dag_id="training_dag",
        conf={"reason": "performance_below_threshold"},
    )

    good_path = model_is_good_log.override(task_id="model_is_good_log")()  # Task ID para o caminho OK

    # 5. LIMPEZA (Recebe o caminho de evaluation_output, que é o resultado de evaluate_production_model)
    cleanup_task = finish_evaluation_job(evaluation_output)

    # 6. DEFINIÇÃO DO FLUXO COMPLETO
    (
            setup_task
            >> evaluation_file_path
            >> processed_eval_path
            >> evaluation_output  # Task de avaliação
            >> decision  # Task de checagem (ramifica)
    )

    # Branching e Convergência
    # O Airflow direciona a execução para uma das tarefas abaixo, e ambas convergem para a limpeza.
    decision >> [trigger_retrain, good_path]

    # A limpeza espera que TODAS as tarefas de decisão/ação tenham terminado.
    [trigger_retrain, good_path] >> cleanup_task