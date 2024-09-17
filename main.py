# -*- coding:utf-8 -*-

import json
import time
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
import subprocess 
from fastapi import APIRouter, HTTPException
from fastapi import BackgroundTasks
from typing import Optional

from fastapi.responses import JSONResponse


import httpx
import schemas
from deps import get_token
from utils import generate_lyrics, generate_music, get_feed, get_lyrics

from datetime import datetime
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from models import Base, User , Song
from database import SessionLocal, engine, get_db
import logging
import requests

import base64
import os
from PIL import Image
from io import BytesIO
import uuid

from manualprompt import initiate_song_generation


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


class DescriptionRequest(BaseModel):
    description: str
    

class DescriptionModeGenerateParam(BaseModel):
    gpt_description_prompt: str
    make_instrumental: bool = False
    mv: str = "chirp-v3-0"

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





# ------------- GET GENERATED TEXT VARIATIONS ----------------
@app.post("/generate-text-variations/")
async def generate_variation_text(payload: dict):
    api_url = 'https://app.riffusion.com/api/trpc/openai.generateTextVariations'

    try:
        # Make the POST request to the external API
        response = requests.post(api_url, json=payload)
        
        # Raise an exception if the request fails
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()

        # Return the response with status code 200
        return JSONResponse(status_code=200, content=data)

    except requests.exceptions.RequestException as e:
        # In case of error, raise an HTTPException with status 500
        raise HTTPException(status_code=500, detail=str(e))

# ------------- GENERATE IMAGE ---------------------
def generate_image(api_url, visual):
    """Generate an image from the visual prompt using the API."""
    form_data = {
        'prompt': (None, visual),  # Use the visual prompt for each request
        'output_format': (None, 'bytes'),  # Request image in bytes
        'user_profile_id': (None, 'null'),
        'anonymous_user_id': (None, '12356e77-a740-432c-bd5a-f151bb8bf16c'),
        'request_timestamp': (None, '1726040383.616'),
        'user_is_subscribed': (None, 'false'),
        'client_id': (None, 'pSgX7WgjukXCBoYwDM8G8GLnRRkvAoJlqa5eAVvj95o')
    }

    response = requests.post(api_url, files=form_data)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch image from API")

    return response.content

def save_image(image_data, output_dir):
    """Save the image data to the specified directory."""
    unique_id = uuid.uuid4()
    filename = f"image_{unique_id}.jpeg"
    output_path = os.path.join(output_dir, filename)
    output_path = os.path.normpath(output_path)

    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    with open(output_path, "wb") as f:
        f.write(image_data)

    return output_path


# ------------- GENERATE IMAGES BASED ON VISUALS ---------------------
@app.post("/create-images/")
async def create_images(payload: dict):
    image_api_url = "https://ai-api.magicstudio.com/api/ai-art-generator"
    output_dir = '/NIBM/music_ai_api/images'  # Directory where images will be saved

    try:
        # Extract visuals from the text variation response
        outputs = payload["result"]["data"]["json"]["outputs"]
        
        # Take only the first two visuals
        visual_prompts = [outputs[0]["visual"], outputs[1]["visual"]]

        # Generate and save images based on the visuals
        saved_files = []
        for visual in visual_prompts:
            image_data = generate_image(image_api_url, visual)
            file_path = save_image(image_data, output_dir)
            saved_files.append(file_path)

        # Return success message with file paths
        return {"status_code": 200, "message": "Images saved successfully", "file_paths": saved_files}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid response structure: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    



# -------------------------------- GENERATE SONG USING SUNO BY DESCRIPTION ---------------------------

class SongDescriptionRequest(BaseModel):
    description: str

def initiate_song_generation(description: str):
    """Initiate the song generation process with the given description and return the clip IDs."""
    url = "http://127.0.0.1:8000/generate/description-mode"
    headers = {
        'Authorization': f'Bearer {os.getenv("AUTH_TOKEN")}',
        'Content-Type': 'application/json',
    }
    data = {
        "gpt_description_prompt": description,
        "make_instrumental": False,
        "mv": "chirp-v3-0"
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        print("--------------------------------------")
        print(response_data)
        print("--------------------------------------")
        if 'clips' in response_data and response_data['clips']:
            clip_ids = [clip['id'] for clip in response_data['clips']]
            return clip_ids
        raise HTTPException(status_code=400, detail="No clips generated or available in response.")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate song generation: {e}")

def get_audio_url_from_clip_id(clip_id: str) -> str:
    """Construct the audio URL from the clip ID."""
    audio_url = f"https://cdn1.suno.ai/{clip_id}.mp3"
    print("----------------------------------")
    print("audio : "+ audio_url)
    print("----------------------------------")
    return audio_url

def download_song(clip_id: str):
    """Download the song from the audio URL in the background and save it to the 'songs' folder."""
    # Construct the audio URL
    audio_url = get_audio_url_from_clip_id(clip_id)
    
    # Define the directory to store the songs
    songs_folder = "songs"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(songs_folder):
        os.makedirs(songs_folder)
    
    # Set headers for the request
    headers = {
        'Authorization': f'Bearer {os.getenv("AUTH_TOKEN")}',
        'Referer': 'https://suno.com',
        'Origin': 'https://suno.com',
    }
    
    # Make the request to download the song
    response = requests.get(audio_url, headers=headers)
    
    if response.status_code == 200:
        # Extract the filename from the audio URL
        filename = audio_url.split('/')[-1]
        
        # Create the full path where the song will be saved
        file_path = os.path.join(songs_folder, filename)
        
        # Save the song to the specified folder
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Song downloaded successfully: {file_path}")
    else:
        print(f"Failed to download the song: {response.status_code} - {response.text}")


@app.post("/generate-song/")
def generate_song(description_request: SongDescriptionRequest, background_tasks: BackgroundTasks):
    """
    Start the song generation process, show the first clip, and download remaining clips in the background.
    :param description_request: SongDescriptionRequest model.
    :param background_tasks: BackgroundTasks to handle downloading in the background.
    """
    description = description_request.description
    clip_ids = initiate_song_generation(description)

    # Wait for a short period to give the server time to process the song
    time.sleep(135)  # Adjust the sleep time as needed

    if not clip_ids:
        raise HTTPException(status_code=400, detail="No clips generated")

    # Process the first clip synchronously and return it
    first_clip_id = clip_ids.pop(0)
    first_filename = download_song(first_clip_id)
    
    # Queue the rest of the clips to be downloaded in the background
    for clip_id in clip_ids:
        background_tasks.add_task(download_song, clip_id)

    return {
        "detail": "Song generation started. The first clip is ready, and the rest will be downloaded in the background.",
        "first_clip": first_filename
    }
# ---------------------------------------------------

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

# ---------------- GENERATE SUNO DESC SONG --------------------

@app.post("/generate/description-mode")
async def generate_with_song_description(
    data: schemas.DescriptionModeGenerateParam, token: str = Depends(get_token)
):
    try:
        # Call the function to generate the music using the description
        resp = await generate_music(data.dict(), token)

        # Return both the response and the token
        return {"response": resp, "token": token}
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ---------------- GENERATE SUNO DESC SONG --------------------


# ------------- ADD SONG -------------------
class SongCreate(BaseModel):
    title: str
    user_song_description: Optional[str]
    custom_lyrics: Optional[str]
    song_type_id: int
    user_id: int

class SongResponse(BaseModel):
    responseMsg: str
    responseCode: str
    responseData: SongCreate

@app.post("/add-songs/", response_model=SongResponse)
def create_song(song: SongCreate, db: Session = Depends(get_db)):
    try:
        # Prepare the data for inserting
        new_song_data = {
            "title": song.title,
            "user_song_description": song.user_song_description,
            "custom_lyrics": song.custom_lyrics,
            "song_type_id": song.song_type_id,
            "created_date": datetime.utcnow(),
            "user_id": song.user_id
        }

        # Create the song record
        db_song = Song(**new_song_data)
        db.add(db_song)
        db.commit()
        db.refresh(db_song)

        # Create a response data object
        song_response_data = {
            "id": db_song.id,
            "title": db_song.title,
            "user_song_description": db_song.user_song_description,
            "custom_lyrics": db_song.custom_lyrics,
            "created_date": db_song.created_date,
            "song_type_id": db_song.song_type_id,
            "user_id": db_song.user_id
        }

        # Return success response
        return SongResponse(responseMsg="Song created", responseCode="200", responseData=song_response_data)

    except HTTPException as http_exc:
        return SongResponse(responseMsg=http_exc.detail, responseCode=http_exc.status_code, responseData=None)
    except Exception as e:
        logging.error(f"Error creating song: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ------------- ADD SONG -------------------

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




@app.post("/suno")
async def generate_song_with_description():
    try:
        # Call the asynchronous function
        result = await initiate_song_generation("A song about schools")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))