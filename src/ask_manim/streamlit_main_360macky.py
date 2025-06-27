import os
import subprocess
import streamlit as st
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import glob

from utils import *

#Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# Initialize Groq-compatible OpenAI client
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

#  Streamlit Page Config
icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
icon = Image.open(icon_path) if os.path.exists(icon_path) else "🎬"
st.set_page_config(page_title="Generative Manim", page_icon=icon)

st.title("Generative Manim")
st.caption("Built by 360macky | Inspired by Grant Sanderson (3Blue1Brown)")
st.write(" Create beautiful mathematical animations from natural language using [Manim](https://docs.manim.community) and LLMs served via Groq.")

#  Prompt Input
prompt = st.text_area(
    " Describe animation (e.g. 'Draw a blue circle and convert it to a red square')",
    placeholder="Draw a blue circle and convert it to a red square"
)

#  Model Selector (Only LLaMA3 supported)
model_choices = {
    "LLaMA3 (70B)": "llama3-70b-8192"
}
model_name = model_choices[st.selectbox("Select Groq model", list(model_choices.keys()))]

generate_video = st.button("🎬 Animate")
show_code = st.checkbox("Show generated code")

#  On Generate
if generate_video:
    if not prompt or len(prompt.strip()) < 10:
        st.error(" Prompt is too short.")
        st.stop()

    clean_prompt = prompt.strip().replace('"', '').replace("'", "").replace("\\", "")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": wrap_prompt(clean_prompt)}
            ],
            temperature=0.7,
            max_tokens=2048
        )
    except Exception as e:
        st.error(f" Groq API Error: {e}")
        st.stop()

    #  Extract & save code
    code_body = extract_construct_code(extract_code(response.choices[0].message.content))
    full_code = create_file_content(code_body)

    if show_code:
        st.text_area(" Generated Code", value=code_body, height=300)

    try:
        with open("GenScene.py", "w") as f:
            f.write(full_code)
    except Exception as e:
        st.error(f" Failed to write GenScene.py: {e}")
        st.stop()

    st.info(" Rendering animation with Manim...")

    try:
        subprocess.run(
            "python -m manim GenScene.py GenScene --format=mp4 --media_dir . --custom_folders video_dir",
            shell=True,
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        st.error(" Manim rendering failed. Try a simpler prompt or check the code.")
        with open("GenScene.py", "rb") as py_file:
            st.download_button("⬇ Download Python file", py_file, "GenScene.py")
        st.stop()

    #  Find and load the generated video
    video_files = glob.glob("**/GenScene.mp4", recursive=True)

    if video_files:
        video_path = video_files[0]
        with open(video_path, "rb") as video_file:
            st.video(video_file.read())
        with open(video_path, "rb") as f:
            st.download_button("⬇ Download Video", f, "GenScene.mp4")
    else:
        st.error(" Couldn't find the rendered video file.")

    with open("GenScene.py", "rb") as py_file:
        st.download_button("⬇ Download Python file", py_file, "GenScene.py")

#  Footer
st.markdown("---")
st.write("Made with using Manim, Groq, and Streamlit")
