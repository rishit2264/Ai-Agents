import streamlit as st
from phi.agent import Agent
from phi.model import GroqChat
from phi.tools.duckduckgo import DuckDuckGo
from pathlib import Path
import tempfile
import time
import os
import gc
import atexit
from dotenv import load_dotenv

# Load Groq key
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit config
st.set_page_config(
    page_title="🎥 Video Summarizer - Groq",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Video Summarizer Agent")
st.header("🚀 Powered by Groq + Mixtral + DuckDuckGo")

# Load model
def get_chat_model():
    return GroqChat(model="mixtral-8x7b-32768", api_key=API_KEY)

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Groq Video Summarizer",
        model=get_chat_model(),
        tools=[DuckDuckGo()],
        markdown=True,
    )

agent = initialize_agent()

video_file = st.file_uploader(
    "Upload a video file", type=["mp4", "mov", "avi"],
    help="The filename will be used for analysis (not the actual video content)."
)

video_path = None

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
        video_name = Path(video_path).name

        # Delete file safely after app exit
        atexit.register(lambda: Path(video_path).unlink(missing_ok=True))

        st.video(video_path, format="video/mp4", start_time=0)

        user_query = st.text_area(
            "What insights are you seeking from the video?",
            placeholder="Ask anything about the video...",
        )

        if st.button("🔍 Analyze Video"):
            if not user_query:
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Thinking with Groq..."):
                        prompt = f"""
The user uploaded a video file named: **{video_name}**.

Use the filename, web context (DuckDuckGo), and the following query to help:

**{user_query}**

If video content is needed but not available, suggest ways the user could extract meaningful information (like transcription).
                        """

                        response = agent.run(prompt)

                    st.subheader("📊 Groq Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"❌ Error during analysis: {error}")
                finally:
                    gc.collect()
else:
    st.info("📁 Upload a video file to get started.")

# Optional: Style text area
st.markdown("""
<style>
.stTextArea textarea {
    height: 100px;
}
</style>
""", unsafe_allow_html=True)
