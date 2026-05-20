from sqlalchemy import Column, Integer, String, Float, ForeignKey, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
import pandas as pd

# Configuración de la base de datos SQLite
DATABASE_URL = "sqlite:///./recommender_new.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelos ORM
class UserDB(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    buy_history = Column(String, nullable=True)
    gustos = Column(String, nullable=True)
    rating_history = relationship('RatingDB', back_populates='user')

class GameDB(Base):
    __tablename__ = 'games'
    game_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    rating_avg = Column(Float, nullable=True)
    no_of_ratings = Column(Integer, nullable=True)
    price = Column(Float, nullable=True)
    categoria = Column(String, nullable=True)
    ratings = relationship('RatingDB', back_populates='game')

class RatingDB(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    game_id = Column(Integer, ForeignKey('games.game_id'))
    rating = Column(Float)
    user = relationship('UserDB', back_populates='rating_history')
    game = relationship('GameDB', back_populates='ratings')

Base.metadata.create_all(bind=engine)

def get_ratings_data():
    db = SessionLocal()
    ratings = db.query(RatingDB).all()
    db.close()
    data = [
        {'userId': r.user_id, 'movieId': r.game_id, 'rating': r.rating}
        for r in ratings
    ]
    return pd.DataFrame(data, columns=['userId', 'movieId', 'rating'])

if __name__ == "__main__":
    db = SessionLocal()
    users = db.query(UserDB).all()
    print("Usuarios en la base de datos:")
    for user in users:
        print(f"ID: {user.user_id}, Username: {user.username}, Gustos: {user.gustos}")
    db.close()