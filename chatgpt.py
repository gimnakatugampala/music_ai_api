import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to get a response from the chat model
def get_chat_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use your desired model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']

# Example usage
if __name__ == "__main__":
    user_input = "What is a LLM?"
    response = get_chat_response(user_input)
    print("Assistant:", response)
