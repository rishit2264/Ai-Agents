import streamlit as st      #Streamlit is a Python framework for building interactive web applications. It allows you to create a user interface with minimal code, perfect for data science and machine learning applications. 
from phi.agent import Agent
from phi.model.google import Gemini 
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file       #These are functions from Google's generative AI library. upload_file lets you upload files to Google's AI services, and get_file retrieves files that have been uploaded.
import google.generativeai as genai

import time
from pathlib import Path

import tempfile     #This module creates temporary files and directories. It's useful for handling uploads or processing files that you only need temporarily.

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


#page configuration
st.set_page_config(
    page_title=("Multimodal AI agent - Video Summarizer"),
    page_icon="🎥",
    layout="wide"
)

st.title("Phi data video ai summarizer Agent")
st.header("powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name = "Video AI Summarizer",
        model = Gemini(id = "gemini-2.0-flash-exp"),
        tools = [DuckDuckGo()],
        markdown=True,
    )

#initialize the agent
multimodal_agent = initialize_agent()

#File uploader
video_file = st.file_uploader(
    "upload a video file",type = ["mp4","mov","avi"] , help= "upload a video for ai analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as temp_video:         #here we are writing the video
        temp_video.write(video_file.read())
        video_path = temp_video.name

        st.video(video_path,format="video/mp4",start_time=0)

        user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )
        if st.button("🔍 Analyze Video",key = "analyze_video_button"):
            if  not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    with st.spinner("processing video and gathering inshights"):         #this will show a spinning during loading
                        #upload and process video file
                        processed_video = upload_file(video_path)                    #this will make a processed video from the video uploaded
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)         #till its showing processing it will keep on uploading the processed video

                        #prompt generation for analysis:
                        analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )
                        
                        #AI agents processing:
                        response = multimodal_agent.run(analysis_prompt,videos=[processed_video])
                    
                    # Display the result
                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                # Clean up temporary video file
                    import gc
                    gc.collect()
                    time.sleep(1)
                    Path(video_path).unlink(missing_ok=True)

else:
    st.info("Upload a video file to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)