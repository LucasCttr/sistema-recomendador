import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from db import get_ratings_data
import pickle
import os

# Evaluación de SVD con validación cruzada

def evaluate_svd_crossval_and_metrics(k=10, threshold=3.5, model_path='svd_surprise.pkl'):
    ratings = get_ratings_data()
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Validación cruzada RMSE y MAE
    model = SVD(n_factors=20, n_epochs=20, reg_all=0.1, random_state=42)
    results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Split para métricas adicionales
    trainset, testset = train_test_split(data, test_size=0.2)

    # Cargar modelo existente o entrenar uno nuevo
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_metrics = pickle.load(f)
    else:
        model_metrics = SVD(n_factors=20, n_epochs=20, reg_all=0.1, random_state=42)
        model_metrics.fit(trainset)

    # Predicciones en test
    test_predictions = model_metrics.test(testset)

    # Precision@K, Recall@K, Coverage
    from collections import defaultdict
    import numpy as np

    user_pred = defaultdict(list)
    for pred in test_predictions:
        user_pred[pred.uid].append((pred.iid, pred.est, pred.r_ui))

    precisions = []
    recalls = []
    recommended_items = set()
    all_items = set(ratings['movieId'].unique())

    for user, preds in user_pred.items():
        sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)
        top_k = sorted_preds[:k]
        n_rel_and_rec_k = sum(1 for (_, _, true_r) in top_k if true_r >= threshold)
        n_rel = sum(1 for (_, _, true_r) in preds if true_r >= threshold)
        if k > 0:
            precisions.append(n_rel_and_rec_k / k)
        if n_rel > 0:
            recalls.append(n_rel_and_rec_k / n_rel)
        for (item, est, _) in top_k:
            if est >= threshold:
                recommended_items.add(item)

    coverage = len(recommended_items) / len(all_items) if len(all_items) > 0 else 0
    precision = np.mean(precisions) if precisions else 0
    recall = np.mean(recalls) if recalls else 0

    return results, precision, recall, coverage

if __name__ == "__main__":
    results, precision, recall, coverage = evaluate_svd_crossval_and_metrics()
    print("\nResultados de validación cruzada:")
    print(f"RMSE promedio: {results['test_rmse'].mean():.4f}")
    print(f"MAE promedio:  {results['test_mae'].mean():.4f}")
    print("\nMétricas de recomendación:")
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10:    {recall:.4f}")
    print(f"Coverage:     {coverage:.4f} ({coverage*100:.2f}% del catálogo)")
