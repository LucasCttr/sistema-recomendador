import pickle
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from db import UserDB, GameDB, RatingDB, SessionLocal
from model import train_surprise_model
from sqlalchemy.orm import Session


MODEL_PATH = 'svd_surprise.pkl'
svd_model = None


def train_and_save_model():
    global svd_model
    svd_model = train_surprise_model()
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(svd_model, f)

app = FastAPI()


@app.on_event("startup")
def regenerate_model_on_startup():
    train_and_save_model()


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


# ── Usuarios ──────────────────────────────────────────────────────────────────

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
        ratings_by_user.setdefault(rating.user_id, []).append({
            "game_id": rating.game_id,
            "game_name": game_name,
            "rating": rating.rating,
        })
    return {"users": [serialize_user(u, ratings_by_user) for u in users]}


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
        id: [{"game_id": r.game_id, "game_name": name, "rating": r.rating}
             for r, name in ratings]
    }
    return serialize_user(db_user, ratings_by_user)


@app.post("/users")
def create_user(user: User, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.user_id == user.user_id).first():
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


# ── Juegos ────────────────────────────────────────────────────────────────────

@app.get("/games")
def list_games(categoria: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(GameDB)
    if categoria:
        query = query.filter(GameDB.categoria == categoria)
    games = query.all()
    return {
        "games": [
            {
                "game_id": g.game_id,
                "name": g.name,
                "categoria": g.categoria,
                "rating_avg": g.rating_avg,
                "no_of_ratings": g.no_of_ratings,
                "price": g.price,
            }
            for g in games
        ]
    }


@app.get("/games/{id}")
def get_game(id: int, db: Session = Depends(get_db)):
    g = db.query(GameDB).filter(GameDB.game_id == id).first()
    if not g:
        raise HTTPException(status_code=404, detail="Game not found")
    return {
        "game_id": g.game_id,
        "name": g.name,
        "categoria": g.categoria,
        "rating_avg": g.rating_avg,
        "no_of_ratings": g.no_of_ratings,
        "price": g.price,
    }


# ── Categorías ────────────────────────────────────────────────────────────────

@app.get("/categorias")
def list_categorias(db: Session = Depends(get_db)):
    rows = db.query(GameDB.categoria).distinct().filter(GameDB.categoria.isnot(None)).all()
    return {"categorias": sorted([r[0] for r in rows])}


@app.get("/categorias/{categoria}/games")
def games_by_categoria(categoria: str, db: Session = Depends(get_db)):
    games = db.query(GameDB).filter(GameDB.categoria == categoria).all()
    if not games:
        raise HTTPException(status_code=404, detail=f"Categoría '{categoria}' no encontrada")
    return {
        "categoria": categoria,
        "total": len(games),
        "games": [
            {
                "game_id": g.game_id,
                "name": g.name,
                "rating_avg": g.rating_avg,
                "no_of_ratings": g.no_of_ratings,
                "price": g.price,
            }
            for g in games
        ]
    }


# ── Recomendaciones ───────────────────────────────────────────────────────────

COLD_START_THRESHOLD = 7

@app.get("/users/{id}/recommend")
def recommend(id: int, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    user_ratings = db.query(RatingDB).filter(RatingDB.user_id == id).all()
    num_ratings = len(user_ratings)
    rated_game_ids = {r.game_id for r in user_ratings}

    buy_history_ids = set()
    if db_user.buy_history and db_user.buy_history.strip():
        buy_history_ids = set(map(int, db_user.buy_history.split(',')))

    excluded_game_ids = buy_history_ids | rated_game_ids

    # COLD START
    if num_ratings < COLD_START_THRESHOLD:
        query = db.query(GameDB)
        if excluded_game_ids:
            query = query.filter(~GameDB.game_id.in_(excluded_game_ids))

        if db_user.gustos:
            query = query.filter(GameDB.categoria == db_user.gustos)
            method = "cold_start_by_gustos"
        else:
            method = "cold_start_popular"

        games = query.order_by(GameDB.rating_avg.desc().nullslast()).limit(10).all()
        return {
            "recommendations": [
                {
                    "game_id": g.game_id,
                    "name": g.name,
                    "categoria": g.categoria,
                    "rating_avg": g.rating_avg,
                    "method": method,
                    "user_ratings_count": num_ratings,
                }
                for g in games
            ]
        }

    # FILTRADO COLABORATIVO
    unbought_games = (
        db.query(GameDB).filter(~GameDB.game_id.in_(excluded_game_ids)).all()
        if excluded_game_ids else db.query(GameDB).all()
    )

    if svd_model is not None:
        predictions = [(g, svd_model.predict(id, g.game_id).est) for g in unbought_games]
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
        return {
            "recommendations": [
                {
                    "game_id": g.game_id,
                    "name": g.name,
                    "categoria": g.categoria,
                    "pred_rating": est,
                    "method": "collaborative_filtering",
                    "user_ratings_count": num_ratings,
                }
                for g, est in recommendations
            ]
        }

    # FALLBACK
    query = db.query(GameDB)
    if excluded_game_ids:
        query = query.filter(~GameDB.game_id.in_(excluded_game_ids))
    games = query.order_by(GameDB.rating_avg.desc().nullslast()).limit(10).all()
    return {
        "recommendations": [
            {
                "game_id": g.game_id,
                "name": g.name,
                "categoria": g.categoria,
                "rating_avg": g.rating_avg,
                "method": "popularity_fallback",
                "user_ratings_count": num_ratings,
            }
            for g in games
        ]
    }


# ── Ratings ───────────────────────────────────────────────────────────────────

@app.post("/users/{id}/rate")
def rate_game(id: int, rating: Rating, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.user_id == id).first()
    db_game = db.query(GameDB).filter(GameDB.game_id == rating.game_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not db_game:
        raise HTTPException(status_code=404, detail="Game not found")

    db.add(RatingDB(user_id=id, game_id=rating.game_id, rating=rating.rating))

    db_game.no_of_ratings = (db_game.no_of_ratings or 0) + 1
    db_game.rating_avg = (
        ((db_game.rating_avg or 0) * (db_game.no_of_ratings - 1) + rating.rating)
        / db_game.no_of_ratings
    )
    db.commit()
    train_and_save_model()
    return {"msg": "Rating added, game average updated, and model updated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)