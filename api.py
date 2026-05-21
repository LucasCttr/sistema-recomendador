import pickle
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

app = FastAPI()


@app.on_event("startup")
def regenerate_model_on_startup():
    train_and_save_model()

# Modelos para validación de datos
class User(BaseModel):
    user_id: int
    username: str
    buy_history: Optional[str] = None
    gustos: Optional[str] = None

class Rating(BaseModel):
    game_id: int
    rating: float


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def serialize_user(user: UserDB, ratings_by_user: dict[int, list[dict]]) -> dict:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "gustos": user.gustos,
        "buy_history": user.buy_history,
        "ratings": ratings_by_user.get(user.user_id, []),
    }


# Endpoint para listar usuarios
@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(UserDB).all()
    ratings = (
        db.query(RatingDB, GameDB.name)
        .join(GameDB, GameDB.game_id == RatingDB.game_id)
        .all()
    )

    ratings_by_user: dict[int, list[dict]] = {}
    for rating, game_name in ratings:
        ratings_by_user.setdefault(rating.user_id, []).append(
            {
                "game_id": rating.game_id,
                "game_name": game_name,
                "rating": rating.rating,
            }
        )

    return {
        "users": [
            serialize_user(user, ratings_by_user)
            for user in users
        ]
    }


# Endpoint para obtener un usuario por ID
@app.get("/users/{id}")
def get_user(id: int, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    ratings = (
        db.query(RatingDB, GameDB.name)
        .join(GameDB, GameDB.game_id == RatingDB.game_id)
        .filter(RatingDB.user_id == id)
        .all()
    )
    ratings_by_user = {
        id: [
            {
                "game_id": rating.game_id,
                "game_name": game_name,
                "rating": rating.rating,
            }
            for rating, game_name in ratings
        ]
    }

    return serialize_user(db_user, ratings_by_user)


# Endpoint para listar juegos
@app.get("/games")
def list_games(db: Session = Depends(get_db)):
    games = db.query(GameDB).all()
    return {
        "games": [
            {
                "game_id": game.game_id,
                "name": game.name,
                "rating_avg": game.rating_avg,
                "no_of_ratings": game.no_of_ratings,
                "price": game.price,
            }
            for game in games
        ]
    }


# Endpoint para obtener un juego por ID
@app.get("/games/{id}")
def get_game(id: int, db: Session = Depends(get_db)):
    db_game = db.query(GameDB).filter(GameDB.game_id == id).first()
    if not db_game:
        raise HTTPException(status_code=404, detail="Game not found")
    return {
        "game_id": db_game.game_id,
        "name": db_game.name,
        "rating_avg": db_game.rating_avg,
        "no_of_ratings": db_game.no_of_ratings,
        "price": db_game.price,
    }

# Endpoint para crear usuario
@app.post("/users")
def create_user(user: User, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == user.user_id).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists")
    db_user = UserDB(
        user_id=user.user_id,
        username=user.username,
        buy_history=user.buy_history,
        gustos=user.gustos,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {
        "msg": "User created",
        "user": {
            "user_id": db_user.user_id,
            "username": db_user.username,
            "gustos": db_user.gustos,
            "buy_history": db_user.buy_history,
        }
    }


# Endpoint para actualizar usuario
@app.put("/users/{id}")
def update_user(id: int, user: User, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.username = user.username
    db_user.buy_history = user.buy_history
    db_user.gustos = user.gustos
    db.commit()
    return {"msg": "User updated"}


# Constante para el threshold de cold start
COLD_START_THRESHOLD = 17  

@app.get("/users/{id}/recommend")
def recommend(id: int, db: Session = Depends(get_db)):
    # 1. Validar que el usuario exista
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 2. Contar cuántos ratings ha hecho el usuario
    user_ratings = db.query(RatingDB).filter(RatingDB.user_id == id).all()
    num_ratings = len(user_ratings)
    rated_game_ids = {rating.game_id for rating in user_ratings}
    
    # 3. Verificar historial de compras para excluir juegos ya adquiridos
    buy_history_ids = set()
    if db_user.buy_history and db_user.buy_history.strip():
        buy_history_ids = set(map(int, db_user.buy_history.split(',')))

    excluded_game_ids = buy_history_ids | rated_game_ids
    
    # --- CASO A: COLD START (Pocos ratings) -> FILTRADO ESTRICTO DESDE SQLITE ---
    if num_ratings < COLD_START_THRESHOLD:
        query = db.query(GameDB)
        
        # Si el usuario tiene un gusto definido, filtramos OBLIGATORIAMENTE por esa categoría en la BD
        if db_user.gustos and db_user.gustos.strip():
            # Esto obliga a SQLite a traer ÚNICAMENTE juegos de esa categoría exacta
            query = query.filter(GameDB.categoria == db_user.gustos.strip())
            
        if excluded_game_ids:
            query = query.filter(~GameDB.game_id.in_(excluded_game_ids))
        
        # Traemos los juegos ordenados por promedio
        games = query.order_by(
            GameDB.rating_avg.desc().nullslast()
        ).limit(10).all()
        
        return {
            "recommendations": [
                {
                    "game_id": g.game_id, 
                    "name": g.name, 
                    "rating_avg": g.rating_avg,
                    "categoria": g.categoria,
                    "method": "cold_start_by_genre_strict",
                    "user_ratings_count": num_ratings
                } 
                for g in games
            ]
        }
    
    # --- CASO B: FILTRADO COLABORATIVO (Suficiente historial) ---
    unbought_games = db.query(GameDB).filter(
        ~GameDB.game_id.in_(excluded_game_ids)
    ).all() if excluded_game_ids else db.query(GameDB).all()
    
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
                    "categoria": g.categoria,
                    "method": "collaborative_filtering",
                    "user_ratings_count": num_ratings
                }
                for g, est in recommendations
            ]
        }
    
    # --- CASO C: FALLBACK (Filtro estricto por categoría si falla el SVD) ---
    query = db.query(GameDB)
    
    if db_user.gustos and db_user.gustos.strip():
        query = query.filter(GameDB.categoria == db_user.gustos.strip())
        
    if excluded_game_ids:
        query = query.filter(~GameDB.game_id.in_(excluded_game_ids))
    
    games = query.order_by(
        GameDB.rating_avg.desc().nullslast()
    ).limit(10).all()
    
    return {
        "recommendations": [
            {
                "game_id": g.game_id, 
                "name": g.name,
                "rating_avg": g.rating_avg,
                "categoria": g.categoria,
                "method": "popularity_genre_fallback_strict",
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