import pickle
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from db import UserDB, GameDB, RatingDB, SessionLocal
from model import train_surprise_model
from sqlalchemy.orm import Session


MODEL_PATH = 'svd_surprise.pkl'
svd_model = None


# Entrenar y guardar el modelo al inicio
def train_and_save_model():
    global svd_model
    svd_model = train_surprise_model()
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(svd_model, f)

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        svd_model = pickle.load(f)
else:
    train_and_save_model()


app = FastAPI()

# Modelos para validación de datos
class User(BaseModel):
    user_id: int
    username: str
    buy_history: Optional[str] = None

class Rating(BaseModel):
    game_id: int
    rating: float


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint para crear usuario
@app.post("/users")
def create_user(user: User, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == user.user_id).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists")
    db_user = UserDB(user_id=user.user_id, username=user.username, buy_history=user.buy_history)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"msg": "User created"}


# Endpoint para actualizar usuario
@app.put("/users/{id}")
def update_user(id: int, user: User, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.username = user.username
    db_user.buy_history = user.buy_history
    db.commit()
    return {"msg": "User updated"}


# Constante para el threshold de cold start (basado en ratings)
COLD_START_THRESHOLD = 7  # Mínimo de ratings para usar filtrado colaborativo

# Endpoint para obtener recomendaciones
@app.get("/users/{id}/recommend")
def recommend(id: int, db: Session = Depends(get_db)):
    # Validar que el usuario exista
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Contar cuántos ratings ha hecho el usuario
    num_ratings = db.query(RatingDB).filter(RatingDB.user_id == id).count()
    
    # Verificar historial de compras para excluir juegos ya comprados
    buy_history_ids = set()
    if db_user.buy_history and db_user.buy_history.strip():
        buy_history_ids = set(map(int, db_user.buy_history.split(',')))
    
    # COLD START: Si tiene menos ratings que el threshold
    if num_ratings < COLD_START_THRESHOLD:
        # Excluir juegos comprados si tiene historial
        query = db.query(GameDB)
        if buy_history_ids:
            query = query.filter(~GameDB.game_id.in_(buy_history_ids))
        
        games = query.order_by(
            GameDB.rating_avg.desc().nullslast()
        ).limit(10).all()
        
        return {
            "recommendations": [
                {
                    "game_id": g.game_id, 
                    "name": g.name, 
                    "rating_avg": g.rating_avg,
                    "method": "cold_start",
                    "user_ratings_count": num_ratings
                } 
                for g in games
            ]
        }
    
    # Usuario con suficiente historial de ratings (>= COLD_START_THRESHOLD)
    # Obtener juegos no comprados por el usuario
    unbought_games = db.query(GameDB).filter(
        ~GameDB.game_id.in_(buy_history_ids)
    ).all() if buy_history_ids else db.query(GameDB).all()
    
    # Si hay modelo entrenado, predecir ratings para juegos no comprados
    if svd_model is not None:
        predictions = []
        for game in unbought_games:
            pred = svd_model.predict(id, game.game_id)
            predictions.append((game, pred.est))
        
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
        return {
            "recommendations": [
                {
                    "game_id": g.game_id, 
                    "name": g.name, 
                    "pred_rating": est,
                    "method": "collaborative_filtering",
                    "user_ratings_count": num_ratings
                }
                for g, est in recommendations
            ]
        }
    
    # Fallback: si no hay modelo, usar rating promedio (excluyendo comprados)
    query = db.query(GameDB)
    if buy_history_ids:
        query = query.filter(~GameDB.game_id.in_(buy_history_ids))
    
    games = query.order_by(
        GameDB.rating_avg.desc().nullslast()
    ).limit(10).all()
    
    return {
        "recommendations": [
            {
                "game_id": g.game_id, 
                "name": g.name,
                "rating_avg": g.rating_avg,
                "method": "popularity_fallback",
                "user_ratings_count": num_ratings
            } 
            for g in games
        ]
    }


# Endpoint para agregar rating   
# Json: {"user_id": 1, "game_id": 2, "rating": 4}
@app.post("/users/{id}/rate")
# Al agregar un nuevo rating, se re-entrena el modelo para mantenerlo actualizado
def rate_game(id: int, rating: Rating, db: Session = Depends(get_db)):
    # Validar que el usuario y el juego existan
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    db_game = db.query(GameDB).filter(GameDB.game_id == rating.game_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not db_game:
        raise HTTPException(status_code=404, detail="Game not found")
    # Agregar el rating a la base de datos
    db_rating = RatingDB(user_id=id, game_id=rating.game_id, rating=rating.rating)
    db.add(db_rating)
    # Actualizar promedio y cantidad de ratings del juego
    if db_game.no_of_ratings is None:
        db_game.no_of_ratings = 0
    if db_game.rating_avg is None:
        db_game.rating_avg = 0.0
    nuevo_total = db_game.no_of_ratings + 1
    nuevo_promedio = (db_game.rating_avg * db_game.no_of_ratings + rating.rating) / nuevo_total
    db_game.no_of_ratings = nuevo_total
    db_game.rating_avg = nuevo_promedio
    db.commit()
    # Re-entrenar y guardar el modelo
    train_and_save_model()
    return {"msg": "Rating added, game average updated, and model updated"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)