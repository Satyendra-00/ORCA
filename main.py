import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="ORCA - Agentic AI")

# Define mock tools
@tool
def get_weather(lat: float, lon: float) -> str:
    """Get weather information for given latitude and longitude."""
    return "Weather alert: 2.5m waves approaching."

@tool
def check_imbl(distance_km: float) -> str:
    """Check if within 5km of IMBL."""
    if distance_km < 5:
        return "DANGER: Within 5km of IMBL. Turn back immediately!"
    else:
        return "Safe. You are well within Indian waters."

@tool
def get_pfz() -> str:
    """Get nearest PFZ location."""
    return "Nearest PFZ is 15km NE. Navigate here for best catch."

# Initialize ChatGroq with a verified active model from your list
llm = ChatGroq(
    model="qwen/qwen3.8-27b", 
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Create the agent
tools = [get_weather, check_imbl, get_pfz]
agent_executor = create_react_agent(llm, tools)

# Define request model
class ChatRequest(BaseModel):
    query: str

# POST endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    # Invoke the agent with the user query as a tuple
    response = agent_executor.invoke({"messages": [("user", request.query)]})
    
    # Extract the final response string
    # The response is a dictionary with a key 'messages' which is a list of messages.
    # We want the content of the last message (the agent's response).
    final_message = response["messages"][-1].content
    return {"response": final_message}