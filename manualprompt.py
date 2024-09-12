
import requests
from dotenv import load_dotenv
import time
import os
import aiohttp
import asyncio

# Load environment variables from .env file
load_dotenv()

def get_audio_url_from_clip_id(clip_id):
	# Directly construct the audio URL from the clip ID
	audio_url = f"https://cdn1.suno.ai/{clip_id}.mp3"
	print(f"Constructed Audio URL: {audio_url}")
	return audio_url

async def initiate_song_generation(description):
    url = "http://127.0.0.1:8000/generate/description-mode"
    headers = {
        'Cookie': f'session_id={os.getenv("SESSION_ID")}; {os.getenv("COOKIE")}',
        'Content-Type': 'application/json'
    }
    data = {
        "gpt_description_prompt": description,
        "make_instrumental": False,
        "mv": "chirp-v3-0"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            print(f"Received response with status: {response.status}")
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                print(f"Error response: {error_text}")
                return {"error": error_text}

def download_song(audio_url):
	headers = {
		'Cookie': f'session_id={os.getenv("SESSION_ID")}; {os.getenv("COOKIE")}'
	}
	response = requests.get(audio_url, headers=headers)
	if response.status_code == 200:
		filename = audio_url.split('/')[-1]
		with open(filename, 'wb') as f:
			f.write(response.content)
		print(f"Song downloaded successfully: {filename}")
	else:
		print("Failed to download the song:", response.status_code, response.text)

def main():
	# description = input("Enter a description for the song you want to generate: ")
	clip_ids = initiate_song_generation("A song about schools")
	if clip_ids:
		print("Waiting for the song to be processed...")
		time.sleep(135)	# Wait for 30 seconds to give the server time to process the song
		for clip_id in clip_ids:
			audio_url = get_audio_url_from_clip_id(clip_id)
			if audio_url:
				download_song(audio_url)
			else:
				print("Failed to retrieve audio URL.")
	else:
		print("Failed to generate song or retrieve clip IDs.")

if __name__ == "__main__":
	main()
