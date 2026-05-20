from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from db import get_ratings_data
from collections import defaultdict
import numpy as np


def evaluate_svd_crossval_and_metrics(k=5, threshold=3.5):
    ratings = get_ratings_data()

    print(f"Total ratings: {len(ratings)}")
    print(f"Usuarios únicos: {ratings['userId'].nunique()}")
    print(f"Juegos únicos: {ratings['movieId'].nunique()}")
    print(f"Ratings >= threshold ({threshold}): {(ratings['rating'] >= threshold).sum()} ({(ratings['rating'] >= threshold).mean()*100:.1f}%)")
    print(f"Avg ratings por usuario: {ratings.groupby('userId').size().mean():.1f}")
    print()

    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # ── 1. Validación cruzada RMSE / MAE ──────────────────────────────────
    model = SVD(n_factors=20, n_epochs=20, reg_all=0.1, random_state=42)
    results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # ── 2. Split para métricas de ranking ─────────────────────────────────
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model_metrics = SVD(n_factors=20, n_epochs=20, reg_all=0.1, random_state=42)
    model_metrics.fit(trainset)

    testset_by_user = defaultdict(list)
    for uid, iid, true_r in testset:
        testset_by_user[uid].append((iid, true_r))

    precisions, recalls = [], []
    recommended_items = set()
    all_items = set(ratings['movieId'].unique())
    usuarios_sin_relevantes = 0

    for uid, items in testset_by_user.items():
        relevant = {iid for iid, true_r in items if true_r >= threshold}

        # Si el usuario no tiene ítems relevantes en el testset, lo saltamos
        if not relevant:
            usuarios_sin_relevantes += 1
            continue

        preds = [(iid, model_metrics.predict(uid, iid).est) for iid, _ in items]

        # Usar min(k, len(items)) para no pedir más recomendaciones que ítems disponibles
        k_efectivo = min(k, len(items))
        top_k = sorted(preds, key=lambda x: x[1], reverse=True)[:k_efectivo]
        hits = sum(1 for iid, _ in top_k if iid in relevant)

        precisions.append(hits / k_efectivo)
        recalls.append(hits / len(relevant))

        for iid, est in top_k:
            if est >= threshold:
                recommended_items.add(iid)

    coverage  = len(recommended_items) / len(all_items) if all_items else 0
    precision = np.mean(precisions) if precisions else 0
    recall    = np.mean(recalls)    if recalls    else 0

    print(f"Usuarios evaluados: {len(precisions)}  |  sin relevantes en testset: {usuarios_sin_relevantes}")

    return results, precision, recall, coverage


if __name__ == "__main__":
    results, precision, recall, coverage = evaluate_svd_crossval_and_metrics(k=5, threshold=3.5)

    print("\nResultados de validación cruzada:")
    print(f"  RMSE promedio : {results['test_rmse'].mean():.4f}")
    print(f"  MAE promedio  : {results['test_mae'].mean():.4f}")

    print("\nMétricas de ranking (k=5, threshold=3.5):")
    print(f"  Precision@5   : {precision:.4f}")
    print(f"  Recall@5      : {recall:.4f}")
    print(f"  Coverage      : {coverage:.4f}  ({coverage*100:.2f}% del catálogo)")