# -*- coding:utf-8 -*-

import json

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware


import schemas
from deps import get_token
from utils import generate_lyrics, generate_music, get_feed, get_lyrics

from datetime import datetime
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from models import Base, User
from database import SessionLocal, engine, get_db
import logging
import requests

import base64
import os
from PIL import Image
from io import BytesIO
import uuid




app = FastAPI()





app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
Base.metadata.create_all(bind=engine)


@app.get("/")
async def get_root():
    return schemas.Response()

# ------------- USER AUTH ------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__ident="2b")

class UserCreate(BaseModel):
    profile_img: str
    first_name: str
    last_name: str
    email: EmailStr
    password: str

class UserSignIn(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    responseMsg: str
    responseCode: str
    responseData : object
    

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

@app.post("/signup/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code="400", detail="Email already registered")
        
        hashed_password = get_password_hash(user.password)

        current_user = {
        "profile_img": user.profile_img,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        }

        db_user = User(
            profile_img = user.profile_img,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            password=hashed_password,
            created_date=datetime.utcnow()
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

      
        
        return UserResponse(responseMsg="User created", responseCode="200",responseData=current_user)
    except HTTPException as http_exc:
        return UserResponse(responseMsg=http_exc.detail, responseCode=http_exc.status_code,responseData=None)
    except Exception as e:
        logging.error(f"Error creating user: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/signin/", response_model=UserResponse)
def sign_in(user: UserSignIn, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if  db_user is None:
            raise HTTPException(status_code="401", detail="User does not exist")
        
        if not db_user or not verify_password(user.password, db_user.password):
            raise HTTPException(status_code="401", detail="Incorrect email or password")
        
        

        current_user = {
            "email": user.email,
            }
        
        return UserResponse(responseMsg="Sign-in successful", responseCode="200",responseData=current_user)
    except HTTPException as http_exc:
        return UserResponse(responseMsg=http_exc.detail, responseCode=http_exc.status_code,responseData=None)
    except Exception as e:
        logging.error(f"Error signing in user: {e}")
        raise HTTPException(status_code="500", detail="Internal Server Error")


@app.post("/google-auth/", response_model=UserResponse)
def google_auth_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        current_user = {
            "profile_img": user.profile_img,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            }
        
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            if not db_user or not verify_password(user.password, db_user.password):
                raise HTTPException(status_code="401", detail="Incorrect email or password")
            
    
            
            return UserResponse(responseMsg="Sign-in successful", responseCode="200",responseData=current_user)
        
        hashed_password = get_password_hash(user.password)
        db_user = User(
            profile_img = user.profile_img,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            password=hashed_password,
            created_date=datetime.utcnow()
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return UserResponse(responseMsg="User created", responseCode="200",responseData=current_user)
    except HTTPException as http_exc:
        return UserResponse(responseMsg=http_exc.detail, responseCode=http_exc.status_code,responseData=None)
    except Exception as e:
        logging.error(f"Error creating user: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ------------- USER AUTH ------------------------

# ------------- GET  GENERATED TEXT  ----------------
@app.post("/generate-text-variations/")
async def generate_variation_text(payload: dict):
    api_url = 'https://app.riffusion.com/api/trpc/openai.generateTextVariations'

    try:
        response = requests.post(api_url, json=payload)  
        response.raise_for_status()
        data = response.json()

        return data

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------- GET  GENERATED TEXT  ---------------

# ------------- GENERATE IMAGE ---------------------
def fetch_images_from_api(api_url, payload):
    response = requests.post(api_url, json=payload)  
    response.raise_for_status()
    data = response.json()
    
    predictions = data['result']['data']['json']['predictions']
    base64_images = [prediction['image'] for prediction in predictions]
    return base64_images



@app.post("/create-image/")
async def fetch_and_save_image(payload: dict):
    api_url = 'https://app.riffusion.com/api/trpc/inference.textToImageBatch'
    output_dir = '/Projects/music_ai_api/images' 

    try:

        # Get the base64 images from engine
        base64_images = fetch_images_from_api(api_url, payload)

        # Save the images
        for i, base64_image in enumerate(base64_images):
            unique_id = uuid.uuid4()
            filename = f"image_{unique_id}.jpeg"
            if base64_image.startswith("data:image/jpeg;base64,"):
                base64_string = base64_image.split("data:image/jpeg;base64,")[1]
            image_data = base64.b64decode(base64_string)
            output_path = os.path.join(output_dir, filename)
            with open(output_path,"wb") as f:
                f.write(image_data)


        return base64_images
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------- GENERATE IMAGE ---------------------


# ------------ GENERATE MUSIC RIFF ------------------
def fetch_music_from_api(api_url, payload):
    response = requests.post(api_url, json=payload)  
    response.raise_for_status()
    data = response.json()
    
    predictions = data['result']['data']['json']['predictions']
    base64_audio = [prediction['audio'] for prediction in predictions]
    return base64_audio

@app.post("/generate-music/")
async def fetch_and_save_music(payload: dict):
    api_url = 'https://app.riffusion.com/api/trpc/inference.textToAudioBatch'
    output_dir = '/Projects/music_ai_api/songs' 

    try:

        # Sanitize Audio
        base64_audio = fetch_music_from_api(api_url, payload)

         # Save the Audio
        for i, base64_image in enumerate(base64_audio):
            unique_id = uuid.uuid4()
            filename = f"song_{unique_id}.mp3"
            if base64_image.startswith("data:audio/mpeg;base64,"):
                base64_string = base64_image.split("data:audio/mpeg;base64,")[1]
            image_data = base64.b64decode(base64_string)
            output_path = os.path.join(output_dir, filename)
            with open(output_path,"wb") as f:
                f.write(image_data)

        return {"message": "Audio saved successfully"}
        # return base64_audio
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------ GENERATE MUSIC RIFF ------------------

@app.post("/generate")
async def generate(
    data: schemas.CustomModeGenerateParam, token: str = Depends(get_token)
):
    try:
        resp = await generate_music(data.dict(), token)
        return resp
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.post("/generate/description-mode")
async def generate_with_song_description(
    data: schemas.DescriptionModeGenerateParam, token: str = Depends(get_token)
):
    try:
        resp = await generate_music(data.dict(), token)
        return resp
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/feed/{aid}")
async def fetch_feed(aid: str, token: str = Depends(get_token)):
    try:
        resp = await get_feed(aid, token)
        return resp
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.post("/generate/lyrics/")
async def generate_lyrics_post(request: Request, token: str = Depends(get_token)):
    req = await request.json()
    prompt = req.get("prompt")
    if prompt is None:
        raise HTTPException(
            detail="prompt is required", status_code=status.HTTP_400_BAD_REQUEST
        )

    try:
        resp = await generate_lyrics(prompt, token)
        return resp
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/lyrics/{lid}")
async def fetch_lyrics(lid: str, token: str = Depends(get_token)):
    try:
        resp = await get_lyrics(lid, token)
        return resp
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
