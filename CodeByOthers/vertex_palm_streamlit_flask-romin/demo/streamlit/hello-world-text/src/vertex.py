

from vertexai.preview.language_models import TextGenerationModel
import vertexai
import streamlit as st

PROJECT_ID = "YOUR_GOOGLE_CLOUD_PROJECT" #Your Google Cloud Project Id
LOCATION_NAME="us-central1" #us-central1 for now
vertexai.init(project=PROJECT_ID, location=LOCATION_NAME)


@st.cache_resource
def get_model():
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    return generation_model


def get_text_generation(prompt="",  **parameters):
    generation_model = get_model()
    response = generation_model.predict(prompt=prompt, **parameters
                                        )

    return response.text


