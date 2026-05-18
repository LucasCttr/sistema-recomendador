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

    # Pesos dinámicos según cantidad de ratings
    # 0 ratings → 100% cold start | muchos ratings → casi 100% SVD
    peso_svd  = num_ratings / (num_ratings + COLD_START_THRESHOLD)
    peso_cold = 1 - peso_svd

    # Juegos candidatos (excluir ya rateados y comprados)
    candidate_games = (
        db.query(GameDB).filter(~GameDB.game_id.in_(excluded_game_ids)).all()
        if excluded_game_ids else db.query(GameDB).all()
    )

    # Filtrar por gustos para el componente cold start si tiene preferencia
    if db_user.gustos:
        gustos_ids = {
            g.game_id for g in db.query(GameDB)
            .filter(GameDB.categoria == db_user.gustos).all()
        }
    else:
        gustos_ids = None

    if svd_model is not None:
        blended = []
        for g in candidate_games:
            pred_svd  = svd_model.predict(id, g.game_id).est
            rating_base = g.rating_avg or 0

            # Si hay gustos, el componente cold start prioriza la categoría preferida
            if gustos_ids is not None:
                cold_score = (rating_base * 1.5) if g.game_id in gustos_ids else rating_base
            else:
                cold_score = rating_base

            score = (peso_svd * pred_svd) + (peso_cold * cold_score)
            blended.append((g, score, pred_svd))

        blended.sort(key=lambda x: x[1], reverse=True)
        top = blended[:10]

        if peso_svd >= 0.5:
            method = "collaborative_filtering"
        elif peso_svd > 0:
            method = "blended"
        else:
            method = "cold_start_by_gustos" if gustos_ids else "cold_start_popular"

        return {
            "recommendations": [
                {
                    "game_id": g.game_id,
                    "name": g.name,
                    "categoria": g.categoria,
                    "score": round(score, 4),
                    "pred_rating_svd": round(pred_svd, 4),
                    "method": method,
                    "user_ratings_count": num_ratings,
                    "peso_svd": round(peso_svd, 2),
                    "peso_cold": round(peso_cold, 2),
                }
                for g, score, pred_svd in top
            ]
        }

    # FALLBACK sin modelo: cold start puro
    query = db.query(GameDB)
    if excluded_game_ids:
        query = query.filter(~GameDB.game_id.in_(excluded_game_ids))
    if db_user.gustos:
        query = query.filter(GameDB.categoria == db_user.gustos)
    games = query.order_by(GameDB.rating_avg.desc().nullslast()).limit(10).all()
    return {
        "recommendations": [
            {
                "game_id": g.game_id,
                "name": g.name,
                "categoria": g.categoria,
                "rating_avg": g.rating_avg,
                "method": "cold_start_by_gustos" if db_user.gustos else "cold_start_popular",
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