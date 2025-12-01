import os
import pickle
import shutil
import sys
import glob
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.decorators import task

AIRFLOW_HOME = '/Users/migueltorikachvili/PycharmProjects/Airflow_ML'

if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from pipeline.preprocess import preprocess
from pipeline.training import  make_test_train, train_logistic, train_randomforest, train_bayes
from pipeline.evaluation import evaluate_accuracy, save_metrics, evaluate_roc_auc, select_production_model

# --- VARIÁVEIS DE AMBIENTE E CAMINHOS ---
DATA_DIR = os.path.join(AIRFLOW_HOME, 'data')
ML_DIR = os.path.join(DATA_DIR, 'projeto-ml')
TRAIN_DIR = os.path.join(ML_DIR, 'train')
TEST_DIR = os.path.join(ML_DIR, 'test')
MODEL_DIR = os.path.join(ML_DIR, 'model')
PROCESSED_DIR = os.path.join(ML_DIR, 'processed')
METRICS_DIR = os.path.join(ML_DIR, 'metrics')
HISTORY_DIR = os.path.join(ML_DIR, 'history')
DEPLOYMENT_DIR = os.path.join(ML_DIR, 'deployment')

with (DAG(
    dag_id="training_dag",
    start_date=datetime(2020, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag):

    @task
    def ensure_local_data_exists():
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(TRAIN_DIR, exist_ok=True)
        os.makedirs(TEST_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        os.makedirs(HISTORY_DIR, exist_ok=True)
        os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
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
    def train_model_logistic():
        model_name = 'LogReg'
        df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_split.csv'))
        x_train = df.drop(['LeaveOrNot'], axis=1)
        y_train = df['LeaveOrNot']
        model = train_logistic(x_train, y_train)
        pickle.dump(model, open(os.path.join(MODEL_DIR, f'model_{model_name}.pkl'), 'wb'))
        return model_name


    @task
    def train_model_randomforest():
        model_name = 'RndForest'
        df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_split.csv'))
        x_train = df.drop(['LeaveOrNot'], axis=1)
        y_train = df['LeaveOrNot']
        model = train_randomforest(x_train, y_train)
        pickle.dump(model, open(os.path.join(MODEL_DIR, f'model_{model_name}.pkl'), 'wb'))
        return model_name


    @task
    def train_model_bayes():
        model_name = 'NaiveBayes'
        df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_split.csv'))
        x_train = df.drop(['LeaveOrNot'], axis=1)
        y_train = df['LeaveOrNot']
        model = train_bayes(x_train, y_train)
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

        # 3. Avaliar a Acurácia e roc auc
        accuracy = evaluate_accuracy(x_test, y_test, model)
        roc_auc = evaluate_roc_auc(x_test, y_test, model)
        print(f"Acurácia do modelo {model_name}: {accuracy:.4f}")
        print(f"ROC AUC do modelo {model_name}: {roc_auc:.4f}")

        # 4. Salvar as Métricas
        save_metrics(METRICS_DIR, model_name, accuracy, roc_auc)

        return model_name


    @task
    def archive_model(model_name):
        # 1. Carregar o timestamp de execução para o nome do arquivo
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Definir caminhos
        original_file_name = f'model_{model_name}.pkl'
        new_file_name = f'model_{model_name}_{current_time}.pkl'

        original_path = os.path.join(MODEL_DIR, original_file_name)
        history_path = os.path.join(HISTORY_DIR, new_file_name)

        # 3. Mover e Renomear o arquivo
        if os.path.exists(original_path):
            shutil.move(original_path, history_path)
            print(f"Modelo {model_name} movido e arquivado em: {history_path}")
        else:
            print(f"AVISO: Arquivo do modelo {original_file_name} não encontrado para arquivamento.")

        # Retorna o nome do modelo para manter o fluxo
        return model_name


    @task(trigger_rule='all_success')
    def select_best_model():
        best_model = select_production_model(METRICS_DIR)

        if best_model:
            print(f"Task de seleção concluída. Modelo '{best_model}' selecionado para deploy.")

        return best_model


    @task
    def deploy_best_model(best_model_name):
        if not best_model_name:
            print("AVISO: Nenhum modelo selecionado para deploy. Pulando o deployment.")
            return

        # 1. Definir o padrão de busca no histórico
        # Ex: procura por 'model_RndForest_*.pkl' em HISTORY_DIR
        target_file_pattern = os.path.join(HISTORY_DIR, f'model_{best_model_name}_*.pkl')
        deployment_path = os.path.join(DEPLOYMENT_DIR, 'model.pkl')

        # 2. Encontrar o arquivo arquivado mais recente
        list_of_files = glob.glob(target_file_pattern)

        if not list_of_files:
            print(f"ERRO: Nenhum arquivo arquivado encontrado para o modelo {best_model_name}.")
            return

        # Pega o arquivo com a data de criação/modificação mais recente (o último arquivado)
        latest_file = max(list_of_files, key=os.path.getctime)

        # 3. Copiar e Renomear
        # Usamos shutil.copy2 para preservar metadados, embora shutil.copy bastaria.
        shutil.copy2(latest_file, deployment_path)

        # Extrai o nome do arquivo arquivado (ex: model_RndForest_20251201_151959.pkl)
        archived_filename = os.path.basename(latest_file)

        # --- LOG DO DEPLOYMENT ---
        log_file_path = os.path.join(METRICS_DIR, 'deployment_log.txt')
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = (
            f"[{current_datetime}] MODELO PROMOVIDO: '{best_model_name}'. "
            f"Arquivo de Origem: {archived_filename}.\n"
        )

        with open(log_file_path, 'a') as f:
            f.write(log_message)

        print(f"✅ Log de deployment salvo em: {log_file_path}")
        # --- FIM DO LOG ---

        print(f"✅ Modelo '{best_model_name}' (Arquivo: {archived_filename}) PROMOVIDO para PRODUÇÃO.")
        print(f"Deployment finalizado em: {deployment_path}")

        return deployment_path


    @task(trigger_rule='all_done')
    def clean_up():
        print("Iniciando a limpeza dos arquivos temporários...")

        files_to_remove = [
            # 1. Arquivo Processado (limpeza da task preprocess_data)
            os.path.join(PROCESSED_DIR, 'train_processed.csv'),

            # 2. Arquivo de Treino após Split (limpeza da task split_data)
            os.path.join(TRAIN_DIR, 'train_split.csv'),

            # 3. Arquivo de Teste (limpeza da task split_data)
            os.path.join(TEST_DIR, 'test.csv'),
        ]

        # Opcional: Arquivo original de treino, se não for mantido
        # files_to_remove.append(os.path.join(TRAIN_DIR, 'train.csv'))

        for file_path in files_to_remove:
            if os.path.exists(file_path):
                # Usamos os.remove para remover arquivos
                os.remove(file_path)
                print(f"✅ Removido: {file_path}")
            else:
                print(f"⚠️ Aviso: Arquivo não encontrado para remoção: {file_path}")

        # Garante que a pasta MODEL_DIR está realmente vazia após o arquivamento
        # Embora a task archive_model já faça a movimentação, esta é uma checagem de segurança.
        if os.path.exists(MODEL_DIR) and not os.listdir(MODEL_DIR):
            print(f"A pasta de modelos ({MODEL_DIR}) está limpa.")

        print("Limpeza concluída.")


    # FLUXO DA DAG

    # 1. Defina as tarefas de treino
    logreg_task = train_model_logistic()
    rndforest_task = train_model_randomforest()
    bayes_task = train_model_bayes()

    # 2. Crie as tarefas de avaliação, passando o output da tarefa de treino
    evaluate_logreg = evaluate_and_log(logreg_task)
    evaluate_rndforest = evaluate_and_log(rndforest_task)
    evaluate_bayes = evaluate_and_log(bayes_task)

    # 3. Crie as tarefas de arquivamento, passando o output da tarefa de avaliação
    archive_logreg = archive_model(evaluate_logreg)
    archive_rndforest = archive_model(evaluate_rndforest)
    archive_bayes = archive_model(evaluate_bayes)

    # 4. Defina a nova tarefa de seleção
    select_model = select_best_model()

    # 5. Defina a nova tarefa de deployment
    deploy_task = deploy_best_model(select_model)  # Passa o nome do melhor modelo retornado

    # 6. Defina a ordem da execução
    (
            ensure_local_data_exists()
            >> load_training_data()
            >> preprocess_data()
            >> split_data()
            >> [logreg_task, rndforest_task, bayes_task]
    )

    # 7. A task de Seleção deve esperar que TODOS os arquivamentos terminem.
    # O Airflow infere as dependências sequenciais: Treino -> Avaliação -> Arquivamento
    [archive_logreg, archive_rndforest, archive_bayes] >> select_model

    # OBS: O Airflow infere as dependências sequenciais:
    # logreg_task >> evaluate_logreg >> archive_logreg
    # E assim por diante para os outros modelos.

    # 8. Fluxo final: Seleção -> Deployment -> Limpeza
    # A task de Limpeza espera o deploy.
    select_model >> deploy_task
    deploy_task >> clean_up()