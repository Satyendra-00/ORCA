import os
import requests
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: API Key nahi mili .env file mein!")
else:
    print("Fetching active models from Groq...\n")
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ ACTIVE MODELS FOR YOUR ACCOUNT:")
        for model in data.get("data", []):
            print(f" - {model['id']}")
    else:
        print(f"Failed to fetch models. Status code: {response.status_code}")