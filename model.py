import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from db import get_ratings_data

# Función para entrenar el modelo colaborativo

def train_surprise_model():
    # Obtener datos de ratings desde la base de datos
    ratings = get_ratings_data()  # Debe devolver un DataFrame con columnas: userId, movieId, rating

    # Configurar el lector de Surprise
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Entrenar modelo SVD con todos los datos
    trainset = data.build_full_trainset()
    model = SVD()
    print("Training Surprise SVD model..." +  str(data.df.shape))
    model.fit(trainset)

    return model

# Ejemplo de función para obtener predicción

def predict_rating(model, user_id, movie_id):
    return model.predict(user_id, movie_id).est
