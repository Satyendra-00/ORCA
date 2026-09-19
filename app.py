import streamlit as st
import requests
import os
import re
import subprocess
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client for Groq's Whisper API
groq_audio_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

st.set_page_config(page_title="ORCA - Fishermen Safety AI", page_icon="🌊", layout="centered")

st.title("🌊 ORCA: Coastal Fishermen Safety Assistant")
st.markdown("Your intelligent, human-like voice companion for safe navigation.")

# UI Columns for Language and Voice Selection
col1, col2 = st.columns(2)
with col1:
    language = st.selectbox(
        "Preferred Language / अपनी भाषा चुनें:",
        ["English", "Hindi (हिन्दी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Bengali (বাংলা)"]
    )
with col2:
    voice_gender = st.selectbox("Assistant Voice / आवाज़:", ["Female", "Male"])

# Mapping for Whisper transcription
lang_code_map = {
    "English": "en", "Hindi (हिन्दी)": "hi", "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te", "Bengali (বাংলা)": "bn"
}
selected_lang_code = lang_code_map.get(language, "hi")

# Mapping for Microsoft Edge Neural TTS Voices
# These are high-quality human-like voices
tts_voice_map = {
    "English": {"Female": "en-IN-NeerjaNeural", "Male": "en-IN-PrabhatNeural"},
    "Hindi (हिन्दी)": {"Female": "hi-IN-SwaraNeural", "Male": "hi-IN-MadhurNeural"},
    "Tamil (தமிழ்)": {"Female": "ta-IN-PallaviNeural", "Male": "ta-IN-ValluvarNeural"},
    "Telugu (తెలుగు)": {"Female": "te-IN-ShrutiNeural", "Male": "te-IN-MohanNeural"},
    "Bengali (বাংলা)": {"Female": "bn-IN-TanishaaNeural", "Male": "bn-IN-BashkarNeural"}
}
selected_tts_voice = tts_voice_map[language][voice_gender]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("---")
st.write("🎙️ **Talk to ORCA:**")
audio = mic_recorder(start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", key='mic')

user_prompt = None

if audio:
    audio_bytes = audio['bytes']
    audio_file_path = "temp_audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(audio_bytes)

    with st.spinner("Listening..."):
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = groq_audio_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language=selected_lang_code
                )
                user_prompt = transcript.text
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
        except Exception as e:
            st.error(f"Transcription error: {e}")

text_prompt = st.chat_input("Or type your question here...")
prompt = user_prompt if user_prompt else text_prompt

def clean_text_for_speech(text):
    """Removes leftover markdown or special characters so the bot doesn't speak them."""
    text = re.sub(r'[*#_]', '', text) # Remove markdown asterisks, hashes, underscores
    text = text.replace('-', ' ') # Replace hyphens with spaces
    text = text.replace(':', '.') # Replace colons with full stops for natural pauses
    return text

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("ORCA is thinking..."):
            try:
                localized_query = f"[Respond strictly in {language} in a natural conversational paragraph. NO MARKDOWN] {prompt}"

                response = requests.post(
                    "http://127.0.0.1:8001/chat",
                    json={"query": localized_query}
                )

                if response.status_code == 200:
                    bot_reply = response.json().get("response", "No response received.")
                else:
                    bot_reply = f"Error from backend: Status code {response.status_code}"

                # Show text on UI
                st.markdown(bot_reply)

                # Clean text and Generate High-Quality Neural Voice
                clean_reply = clean_text_for_speech(bot_reply)
                tts_file_path = "response_audio.mp3"

                # Using Edge-TTS via subprocess for synchronous execution in Streamlit
                try:
                    # Get the absolute path to edge-tts in the virtual environment
                    edge_tts_path = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "edge-tts.exe")
                    # If not found, try without the virtual environment (fallback to PATH)
                    if not os.path.exists(edge_tts_path):
                        edge_tts_path = "edge-tts"

                    subprocess.run([
                        edge_tts_path,
                        "--voice", selected_tts_voice,
                        "--text", clean_reply,
                        "--write-media", tts_file_path
                    ], check=True)

                    # Play the humanized audio
                    st.audio(tts_file_path, format="audio/mp3", autoplay=True)
                except Exception as tts_err:
                    st.warning(f"Could not generate neural voice output: {tts_err}")

                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"Could not connect to backend. Error: {e}")