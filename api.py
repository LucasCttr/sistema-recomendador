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
    user_id: int
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


# Endpoint para obtener recomendaciones
@app.get("/users/{id}/recommend")
# Al obtener recomendaciones, se utiliza el modelo entrenado para predecir ratings de juegos no calificados por el usuario y se devuelven los mejores 10
def recommend(id: int, db: Session = Depends(get_db)):
    # Validar que el usuario exista
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    # Obtener ratings calificados por el usuario
    user_ratings = db.query(RatingDB).filter(RatingDB.user_id == id).all()
    # Obtener IDs de juegos calificados por el usuario
    rated_game_ids = {r.game_id for r in user_ratings}
    # Obtener juegos no calificados por el usuario
    unrated_games = db.query(GameDB).filter(~GameDB.game_id.in_(rated_game_ids)).all()
    # Si hay modelo entrenado, predecir ratings para juegos no calificados y devolver los mejores 10
    if svd_model is not None:
        predictions = []
        # Predecir rating para cada juego no calificado
        for game in unrated_games:
            # Predecir rating usando el modelo SVD de Surprise
            pred = svd_model.predict(id, game.game_id)
            # Guardar la predicción junto con el juego para ordenar posteriormente
            predictions.append((game, pred.est))
            # Ordenar por rating predicho y tomar los mejores 10
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
        return {
            "recommendations": [
                {"game_id": g.game_id, "name": g.name, "pred_rating": est}
                for g, est in recommendations
            ]
        }
    # Fallback: lógica mock si no hay modelo
    games = db.query(GameDB).order_by(GameDB.rating_avg.desc().nullslast()).limit(10).all()
    return {"recommendations": [{"game_id": g.game_id, "name": g.name} for g in games]}


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
    db.commit()
    # Re-entrenar y guardar el modelo
    train_and_save_model()
    return {"msg": "Rating added and model updated"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)