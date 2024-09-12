from fastapi import APIRouter
from pydantic import BaseModel
import requests
import os

router = APIRouter()

class DescriptionModeGenerateParam(BaseModel):
    gpt_description_prompt: str
    make_instrumental: bool = False
    mv: str = "chirp-v3-0"

@router.post("/generate-description")
async def generate_description(data: DescriptionModeGenerateParam):
    url = "http://127.0.0.1:8000/generate/description-mode"
    headers = {
        'Cookie': f'session_id={os.getenv("SESSION_ID")}; {os.getenv("COOKIE")}'
    }
    response = requests.post(url, json=data.dict(), headers=headers)
    return response.json()
