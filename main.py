import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import requests

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="ORCA - Agentic AI")

# Add CORS middleware to allow all origins, credentials, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define mock tools with VERY STRICT descriptions so the AI never misses them
@tool
def get_weather(lat: float, lon: float) -> str:
    """Always use this tool to get weather information, alerts, or wave heights for any given location."""
    try:
        # Fetch current weather (temperature and wind speed)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # Fetch marine data (wave height)
        marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height"
        marine_response = requests.get(marine_url)
        marine_response.raise_for_status()
        marine_data = marine_response.json()

        # Extract values
        temperature = weather_data.get('current', {}).get('temperature_2m')
        wind_speed = weather_data.get('current', {}).get('wind_speed_10m')
        wave_height = marine_data.get('current', {}).get('wave_height')

        # Check if we got the required data
        if temperature is None or wind_speed is None:
            raise ValueError("Missing temperature or wind speed data")

        # Format the response string
        return f"Current weather at location: Temp {temperature}°C, Wind {wind_speed} km/h, Wave height {wave_height} meters."

    except Exception as e:
        # Fallback string
        return f"Unable to fetch real-time weather data: {str(e)}. Please check the location or try again later."

@tool
def check_imbl(distance_km: float) -> str:
    """Always use this tool to check if a location is safe or within the 5km IMBL danger zone."""
    if distance_km < 5:
        return "DANGER: Within 5km of IMBL. Turn back immediately!"
    else:
        return "Safe. You are well within Indian waters."

@tool
def get_pfz() -> str:
    """ALWAYS use this tool when the user asks about 'fishing zones', 'where to catch fish', 'highly potential fishing zones', or 'PFZ'."""
    return "Nearest Highly Potential Fishing Zone (PFZ) is 15km North-East. Abundant catch expected today."

# Initialize ChatGroq
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# STRICT SYSTEM PROMPT - Conversational Mode
system_prompt = """You are ORCA, an advanced maritime safety AI voice assistant for fishermen.
NEVER say you do not have access to real-time data. You have tools to check weather, IMBL safety, and Potential Fishing Zones (PFZ).
Whenever a user asks about catching fish or fishing zones, MUST use the get_pfz tool.
CRITICAL RULE FOR OUTPUT: Your response will be spoken aloud by a text-to-speech engine.
- DO NOT use markdown formatting (no asterisks, no bold, no hashes).
- DO NOT use bullet points or numbered lists.
- DO NOT use special characters like hyphens or colons.
- Write purely in natural, conversational, human-like sentences.
- Use commas and full stops for natural pausing.
- Answer concisely and confidently in a conversational tone."""

# Create the agent WITHOUT any extra kwargs to completely avoid version errors
tools = [get_weather, check_imbl, get_pfz]
agent = create_react_agent(llm, tools)

# Define request model
class ChatRequest(BaseModel):
    query: str

# POST endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    # Pass the system prompt directly in the messages array at runtime (100% foolproof)
    response = agent.invoke({
        "messages": [
            ("system", system_prompt),
            ("user", request.query)
        ]
    })

    # Extract the final response string
    final_message = response["messages"][-1].content
    return {"response": final_message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)