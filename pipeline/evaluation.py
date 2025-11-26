from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
import os


def evaluate_accuracy(x_test, y_test, model):
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def save_metrics(metrics_dir, model_name, accuracy):
    """
    Salva as métricas em um arquivo CSV, criando-o ou anexando a nova linha.
    """
    metrics_path = os.path.join(metrics_dir, 'metrics.csv')

    # Prepara a nova linha de métricas
    new_metric = pd.DataFrame([{
        'model_name': model_name,
        'training_date': datetime.now().isoformat(sep=' ', timespec='seconds'),
        'accuracy': accuracy
    }])

    # Verifica se o arquivo já existe para anexar, ou cria um novo
    if os.path.exists(metrics_path):
        existing_metrics = pd.read_csv(metrics_path)
        updated_metrics = pd.concat([existing_metrics, new_metric], ignore_index=True)
        updated_metrics.to_csv(metrics_path, index=False)
    else:
        new_metric.to_csv(metrics_path, index=False)

    print(f"Métricas salvas com sucesso em: {metrics_path}")