from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from datetime import datetime
import os


def evaluate_accuracy(x_test, y_test, model):
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def evaluate_roc_auc(x_test, y_test, model):
    # Use as probabilidades (se o modelo suportar) para um cálculo de ROC AUC mais preciso
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        # Fallback para as classes (como você fez)
        y_score = model.predict(x_test)

    return roc_auc_score(y_test, y_score)


def save_metrics(metrics_dir, model_name, accuracy, roc_auc):
    """
    Salva as métricas em um arquivo CSV, criando-o ou anexando a nova linha.
    """
    metrics_path = os.path.join(metrics_dir, 'train_metrics.csv')

    # Prepara a nova linha de métricas
    new_metric = pd.DataFrame([{
        'model_name': model_name,
        'training_date': datetime.now().isoformat(sep=' ', timespec='seconds'),
        'accuracy': accuracy,
        'roc_auc': roc_auc,
    }])

    # Verifica se o arquivo já existe para anexar, ou cria um novo
    if os.path.exists(metrics_path):
        existing_metrics = pd.read_csv(metrics_path)
        updated_metrics = pd.concat([existing_metrics, new_metric], ignore_index=True)
        updated_metrics.to_csv(metrics_path, index=False)
    else:
        new_metric.to_csv(metrics_path, index=False)

    print(f"Métricas salvas com sucesso em: {metrics_path}")


def select_production_model(metrics_dir):
    metrics_path = os.path.join(metrics_dir, 'train_metrics.csv')

    if not os.path.exists(metrics_path):
        print("Erro: Arquivo de métricas não encontrado. Não é possível selecionar o modelo.")
        return None

    df = pd.read_csv(metrics_path)

    # 1. NOVO FILTRO: Garante que a coluna de data é um datetime para ordenação
    df['training_date'] = pd.to_datetime(df['training_date'])

    # Assumimos que a última rodada da DAG corresponde aos 3 registros mais recentes
    num_models = 3  # Número de modelos treinados

    latest_run_df = df.sort_values(
        by='training_date',
        ascending=False
    ).head(num_models).copy()  # <--- Pega as 3 linhas mais recentes

    if latest_run_df.empty:
        print("Aviso: Nenhuma métrica encontrada para comparação.")
        return None

    # 2. Regra de Comparação: Ordenar por ROC AUC (primário) e Acurácia (secundário)
    best_model_row = latest_run_df.sort_values(
        by=['roc_auc', 'accuracy'],
        ascending=[False, False]  # Mantém a lógica correta (Decrescente)
    ).iloc[0]

    best_model_name = best_model_row['model_name']
    latest_ts_log = latest_run_df['training_date'].max().strftime('%Y-%m-%d %H:%M:%S')  # Log do timestamp mais recente

    print(f"--- Decisão do Modelo para Produção ---")
    print(f"Data da Rodada (Base Máxima): {latest_ts_log}")
    print(f"O MELHOR MODELO SELECIONADO é: {best_model_name}")
    print(f"Métricas Vencedoras: ROC AUC={best_model_row['roc_auc']:.4f}, Accuracy={best_model_row['accuracy']:.4f}")

    return best_model_name


def log_production_metrics(metrics_dir, accuracy, roc_auc):
    """Salva a performance do modelo em produção no log de avaliação."""
    log_path = os.path.join(metrics_dir, 'production_evaluation_metrics.csv')

    new_entry = pd.DataFrame([{
        'evaluation_date': datetime.now().isoformat(sep=' ', timespec='seconds'),
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }])

    if os.path.exists(log_path):
        existing_logs = pd.read_csv(log_path)
        updated_logs = pd.concat([existing_logs, new_entry], ignore_index=True)
        updated_logs.to_csv(log_path, index=False)
    else:
        new_entry.to_csv(log_path, index=False)

    print(f"Produção de métricas de avaliação salvas em: {log_path}")