from vertexai.preview.language_models import TextGenerationModel,TextEmbeddingModel
import vertexai
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_random_exponential

PROJECT_ID = ""
vertexai.init(project=PROJECT_ID, location="us-central1")


@st.cache_resource
def get_model():
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    return generation_model

@st.cache_resource
def get_embedding_model():
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    return embedding_model



def get_text_generation(prompt="",  **parameters):
    generation_model = get_model()
    response = generation_model.predict(prompt=prompt, **parameters
                                        )

    return response.text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def text_generation_model_with_backoff(**kwargs):
    return get_model().predict(**kwargs).text


@retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(5))
def embedding_model_with_backoff(text=[]):
    embeddings = get_embedding_model().get_embeddings(text)
    return [each.values for each in embeddings][0]