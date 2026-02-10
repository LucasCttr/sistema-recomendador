
from sqlalchemy import Column, Integer, String, Float, ForeignKey, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
import pandas as pd

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



# Función para obtener todos los ratings como DataFrame
def get_ratings_data():
    """
    Devuelve un DataFrame con los ratings de la base de datos.
    Columnas: userId, movieId, rating
    """
    db = SessionLocal()
    ratings = db.query(RatingDB).all()
    db.close()
    data = [
        {'userId': r.user_id, 'movieId': r.game_id, 'rating': r.rating}
        for r in ratings
    ]
    return pd.DataFrame(data, columns=['userId', 'movieId', 'rating'])



# # Función para poblar la base de datos con usuarios de ejemplo
# def populate_users():
#     """
#     Inserta 20 usuarios de ejemplo en la tabla users.
#     """
#     db = SessionLocal()
#     for i in range(1, 21):
#         user = UserDB(
#             user_id=i,
#             username=f"user{i}",
#             buy_history=None
#         )
#         db.add(user)
#     db.commit()
#     db.close()

# # Función para poblar la base de datos con ratings de ejemplo
# import random
# def populate_ratings():
#     """
#     Inserta ratings aleatorios para los usuarios y juegos existentes.
#     Cada usuario califica 10 juegos aleatorios con ratings entre 1 y 5.
#     """
#     db = SessionLocal()
#     users = db.query(UserDB).all()
#     games = db.query(GameDB).all()
#     for user in users:
#         rated_games = random.sample(games, min(10, len(games)))
#         for game in rated_games:
#             rating = RatingDB(
#                 user_id=user.user_id,
#                 game_id=game.game_id,
#                 rating=round(random.uniform(1, 5), 1)
#             )
#             db.add(rating)
#     db.commit()
#     db.close()


# Función para poblar la base de datos con 100 juegos reales
# def populate_games():
#     """
#     Inserta 100 juegos de ejemplo en la tabla games.
#     """
#     games_list = [
#         # Juegos populares, puedes modificar o ampliar la lista
#         "Minecraft", "Grand Theft Auto V", "The Legend of Zelda: Breath of the Wild", "Red Dead Redemption 2",
#         "Fortnite", "The Witcher 3: Wild Hunt", "Super Mario Odyssey", "Call of Duty: Modern Warfare",
#         "Overwatch", "Cyberpunk 2077", "Among Us", "Animal Crossing: New Horizons", "Dark Souls III",
#         "Halo Infinite", "FIFA 22", "League of Legends", "Counter-Strike: Global Offensive", "Dota 2",
#         "Apex Legends", "Valorant", "Pokémon Sword", "Pokémon Shield", "Assassin's Creed Valhalla",
#         "God of War", "Spider-Man", "Battlefield V", "PUBG", "Rocket League", "Fall Guys",
#         "Genshin Impact", "Monster Hunter: World", "Sekiro: Shadows Die Twice", "Resident Evil Village",
#         "Death Stranding", "Far Cry 6", "Horizon Zero Dawn", "Destiny 2", "Rainbow Six Siege",
#         "Elden Ring", "Splatoon 2", "Super Smash Bros. Ultimate", "Mario Kart 8 Deluxe",
#         "Persona 5", "Final Fantasy VII Remake", "Star Wars Jedi: Fallen Order", "Tetris Effect",
#         "Cuphead", "Celeste", "Hades", "Stardew Valley", "Terraria", "Portal 2", "Half-Life: Alyx",
#         "DOOM Eternal", "Control", "Metro Exodus", "No Man's Sky", "The Sims 4", "NBA 2K22",
#         "Madden NFL 22", "Forza Horizon 5", "Gran Turismo 7", "Little Nightmares II", "It Takes Two",
#         "Returnal", "Ratchet & Clank: Rift Apart", "Ghost of Tsushima", "Days Gone", "Bloodborne",
#         "Uncharted 4", "The Last of Us Part II", "Detroit: Become Human", "Until Dawn", "Heavy Rain",
#         "Fire Emblem: Three Houses", "Xenoblade Chronicles 2", "Dragon Quest XI", "Yakuza: Like a Dragon",
#         "Persona 4 Golden", "Disco Elysium", "Divinity: Original Sin 2", "Baldur's Gate 3",
#         "Path of Exile", "Torchlight II", "Diablo III", "World of Warcraft", "StarCraft II",
#         "Age of Empires II", "Sid Meier's Civilization VI", "SimCity", "Cities: Skylines",
#         "Planet Zoo", "Zoo Tycoon", "RollerCoaster Tycoon", "The Forest", "Subnautica",
#         "ARK: Survival Evolved", "Rust", "Valheim", "Don't Starve", "Slay the Spire"
#     ]
#     db = SessionLocal()
#     for idx, name in enumerate(games_list, start=1):
#         game = GameDB(
#             game_id=idx,
#             name=name,
#             rating_avg=None,
#             no_of_ratings=0,
#             price=round(10 + idx * 0.5, 2)  # Precio ejemplo
#         )
#         db.add(game)
#     db.commit()
#     db.close()



if __name__ == "__main__":
    db = SessionLocal()
    users = db.query(UserDB).all()
    print("Usuarios en la base de datos:")
    for user in users:
        print(f"ID: {user.user_id}, Username: {user.username}, Buy history: {user.buy_history}")
    db.close()
    
    
    
    # Poblar la base de datos con juegos, usuarios y ratings
    # Descomenta las siguientes líneas para poblar la base de datos:
    # populate_games()
    # populate_users()
    # populate_ratings()
