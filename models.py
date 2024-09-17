from sqlalchemy import Boolean, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    profile_img = Column(String, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_date = Column(DateTime, default=datetime.utcnow)
    disabled = Column(Boolean, default=False)

class Song(Base):
    __tablename__ = "generated_song"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    user_song_description = Column(Text, nullable=True)
    custom_lyrics = Column(Text, nullable=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    song_type_id = Column(Integer)
    user_id = Column(Integer)
