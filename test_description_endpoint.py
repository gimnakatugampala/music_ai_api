import json
import requests

def test_generate_music_with_description():
    data = {
        "gpt_description_prompt": "A Blues song about a person who is feeling happy and optimistic about the future.",
        "make_instrumental": False,
        "mv": "chirp-v3-0",
    }

    url = "http://127.0.0.1:8000/generate/description-mode"
    response = requests.post(url, data=json.dumps(data))
    
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print(f"Failed to generate song description: {response.status_code} {response.text}")

test_generate_music_with_description()
