import pickle
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from db import UserDB, GameDB, RatingDB, SessionLocal
from sqlalchemy.orm import Session


MODEL_PATH = 'svd_surprise.pkl'
svd_model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        svd_model = pickle.load(f)


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


@app.put("/users/{id}")
def update_user(id: int, user: User, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.username = user.username
    db_user.buy_history = user.buy_history
    db.commit()
    return {"msg": "User updated"}


@app.get("/users/{id}/recommend")
def recommend(id: int, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    user_ratings = db.query(RatingDB).filter(RatingDB.user_id == id).all()
    rated_game_ids = {r.game_id for r in user_ratings}
    unrated_games = db.query(GameDB).filter(~GameDB.game_id.in_(rated_game_ids)).all()
    if svd_model is not None:
        predictions = []
        for game in unrated_games:
            pred = svd_model.predict(id, game.game_id)
            predictions.append((game, pred.est))
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
        return {
            "recommendations": [
                {"game_id": g.game_id, "name": g.name, "pred_rating": est}
                for g, est in recommendations
            ]
        }
    # Fallback: lógica mock si no hay modelo
    games = db.query(GameDB).order_by(GameDB.rating_avg.desc().nullslast()).limit(5).all()
    return {"recommendations": [{"game_id": g.game_id, "name": g.name} for g in games]}


# Endpoint para agregar rating
@app.post("/users/{id}/rate")
def rate_game(id: int, rating: Rating, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    db_game = db.query(GameDB).filter(GameDB.game_id == rating.game_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not db_game:
        raise HTTPException(status_code=404, detail="Game not found")
    db_rating = RatingDB(user_id=id, game_id=rating.game_id, rating=rating.rating)
    db.add(db_rating)
    db.commit()
    return {"msg": "Rating added"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)