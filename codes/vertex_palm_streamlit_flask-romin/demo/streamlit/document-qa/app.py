import streamlit as st
from vertexai.preview.language_models import TextGenerationModel
import vertexai
from PIL import Image
from src.utils import *
from src.vertex import *
from streamlit_chat import message
from io import StringIO



st.set_page_config(
    page_title="Question Answering using Vertex PaLM API",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This app shows you how to use Vertex PaLM Text Generator API on your custom documents"
    }
)

#creating session states    
create_session_state()



image = Image.open('./image/palm.jpg')
st.image(image)
st.title(":red[QA Bot] :blue[with PaLM API] on :green[documents]")
st.markdown("<h5 style='text-align: center; color: darkblue;'>Engine: Native python implementation of vector store and text-bison@001.</h5>", unsafe_allow_html=True)
@st.cache_data
def cache_vector_store(data):
    vector_store = data.copy()
    return vector_store

with st.sidebar:
    user_docs = st.file_uploader(
            "Upload your documents here and click on 'Process'", accept_multiple_files=True)
    
    chunk_size_value = st.number_input("Chunk Size:",value=500,min_value=200, max_value=10000,step=200)
    st.session_state['chunk_size'] = chunk_size_value

    sample_bool_val = st.radio("Sample Vector Store?", (True, False))
    st.session_state['sample_bool'] = sample_bool_val
    
    if st.session_state['sample_bool']:
        sample_value = st.number_input("Sample Size:",value=10,min_value=1, max_value=10,step=1)
        st.session_state['sample_value'] = sample_value

    top_match_result_value = st.number_input("How many top N result to pick?",value=3)
    st.session_state['top_sort_value'] = top_match_result_value

    if not st.session_state['process_doc']:
        if st.button("Process"):
            st.session_state['process_doc'] = True
            with st.spinner("Processing...this will take time..."):
                final_data = read_documents(user_docs,
                                            chunk_size_value=st.session_state['chunk_size'],
                                            sample=st.session_state['sample_bool'],
                                             sample_size=st.session_state['sample_value'])
                # vector_store = cache_vector_store(final_data)
                st.session_state['vector_store'] = final_data
    else:
        st.write("Your vector store is already built. Here's the shape of it:")
        st.write(st.session_state['vector_store'].shape)
    
    if st.button("Relode/Reprocess files"):
        st.session_state['process_doc'] = False

    if st.button("Reset Session"):
        reset_session()

with st.container():
    if not st.session_state['vector_store'].empty:
        # st.write(st.session_state['vector_store'])
        question = st.text_input('What would you like to ask the documents?')
        # get the custom relevant chunks from all the chunks in vector store.
        if question:
            context, top_matched_df, source = get_context_from_question(
                question,
                vector_store=st.session_state['vector_store'],
                sort_index_value=st.session_state['top_sort_value'],  # Top N results to pick from embedding vector search
            )
            prompt = f""" Answer the question as precise as possible using the provided context. \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                """
            
            st.markdown("<h3 style='text-align: center; color: black;'>Here's the answer from document</h3>", unsafe_allow_html=True)
            # st.write("Here's the answer from document: ")
            st.write(get_text_generation(prompt=prompt))
            # st.write("Here's the source from document: ")
            st.markdown("<h4 style='text-align: center; color: black;'>Here's the source from document:</h4>", unsafe_allow_html=True)
            st.dataframe(top_matched_df)
    else:
        st.write("Your vector is not created")
