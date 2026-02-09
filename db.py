from sqlalchemy import Column, Integer, String, Float, ForeignKey, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

# Configuración de la base de datos SQLite
DATABASE_URL = "sqlite:///./recommender.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelos ORM
class UserDB(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    buy_history = Column(String, nullable=True)  # Puede ser JSON o texto
    rating_history = relationship('RatingDB', back_populates='user')

class GameDB(Base):
    __tablename__ = 'games'
    game_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    rating_avg = Column(Float, nullable=True)
    no_of_ratings = Column(Integer, nullable=True)
    price = Column(Float, nullable=True)
    ratings = relationship('RatingDB', back_populates='game')

class RatingDB(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    game_id = Column(Integer, ForeignKey('games.game_id'))
    rating = Column(Float)
    user = relationship('UserDB', back_populates='rating_history')
    game = relationship('GameDB', back_populates='ratings')


# Crear tablas
Base.metadata.create_all(bind=engine)

# Bloque para imprimir usuarios
if __name__ == "__main__":
    db = SessionLocal()
    users = db.query(UserDB).all()
    print("Usuarios en la base de datos:")
    for user in users:
        print(f"ID: {user.user_id}, Username: {user.username}, Buy history: {user.buy_history}")
    db.close()
