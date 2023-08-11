import streamlit as st
from streamlit_chat import message
import pandas as pd
import ast
from PyPDF2 import PdfReader
import textract
import os
import re
from src.vertex import *
import numpy as np

def reset_session() -> None:
    """_summary_: Resets the session state to default values.
    """
    st.session_state['temperature'] = 0.0
    st.session_state['token_limit'] = 256
    st.session_state['top_k'] = 40
    st.session_state['top_p'] = 0.8
    st.session_state['debug_mode'] = False
    st.session_state['prompt'] = []
    st.session_state['response'] = []
    st.session_state['vector_store'] = pd.DataFrame()
    st.session_state['process_doc'] = False
    st.session_state['chunk_size'] = 500
    st.session_state['sample_bool'] = True
    st.session_state['sample_value'] = 10
    st.session_state['top_sort_value'] = 5


def hard_reset_session() -> None: 
    st.session_state = {states : [] for states in st.session_state}


def create_session_state():
    """
    Creating session states for the app.
    """
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.0
    if 'token_limit' not in st.session_state:
        st.session_state['token_limit'] = 256
    if 'top_k' not in st.session_state:
        st.session_state['top_k'] = 40
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.8
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = []
    if 'response' not in st.session_state:
        st.session_state['response'] = []
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = pd.DataFrame()
    if 'process_doc' not in st.session_state:
        st.session_state['process_doc'] = False
    if 'chunk_size' not in st.session_state:
        st.session_state['chunk_size'] = 500
    if 'sample_bool' not in st.session_state:
        st.session_state['sample_bool'] = True
    if 'sample_value' not in st.session_state:
        st.session_state['sample_value'] = 10
    if 'top_sort_value' not in st.session_state:
        st.session_state['top_sort_value'] = 5


    
def create_data_packet(file_name, file_type, page_number, file_content):
    """Creating a simple dictionary to store all information (content and metadata)
    extracted from the document"""
    data_packet = {}
    data_packet["file_name"] = file_name
    data_packet["file_type"] = file_type
    data_packet["page_number"] = page_number
    data_packet["content"] = file_content
    return data_packet

def get_chunks_iter(text, maxlength):
    """
    Get chunks of text, each of which is at most maxlength characters long.

    Args:
        text: The text to be chunked.
        maxlength: The maximum length of each chunk.

    Returns:
        An iterator over the chunks of text.
    """
    start = 0
    end = 0
    final_chunk = []
    while start + maxlength < len(text) and end != -1:
        end = text.rfind(" ", start, start + maxlength + 1)
        final_chunk.append(text[start:end])
        start = end + 1
    final_chunk.append(text[start:])
    return final_chunk


# function to apply "get_chunks_iter" function on each row of dataframe.
# currently each row here for file_type=pdf is content of each page and for other file_type its the whole document.
def split_text(row):
    """_summary_: Splits the text into chunks of given size.

    Args:
        row (_type_): each row of the pandas dataframe through apply function.

    Returns:
        _type_: list of chunks of text.
    """
    chunk_iter = get_chunks_iter(row, chunk_size)
    return chunk_iter


@st.cache_data
def read_documents(documents,chunk_size_value=2000, sample=True, sample_size=10):
    """_summary_: Reads the documents and creates a pandas dataframe with all the content and metadata.
    cleaning the text and splitting the text into chunks of given size. creating a vector store of the chunks.

    Args:
        documents (_type_): list of documents uploaded by the user.
        chunk_size_value (_type_, optional): size of each chunk. Defaults to 2000.
        sample (bool, optional): whether to create a sample vector store or not. Defaults to True.
        sample_size (int, optional): size of the sample vector store. Defaults to 10.

    Returns:
        _type_: pandas dataframe with all the content and metadata.
    
    """
    final_data = []
    with st.spinner('Loading documents and putting them in pandas dataframe.....'):
        for eachdoc in documents:
            file_name, file_type = os.path.splitext(eachdoc.name)
            if file_type == ".pdf":
                # loading pdf files, with page numbers as metadata.
                reader = PdfReader(eachdoc)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        packet = create_data_packet(
                            file_name, file_type, page_number=int(i + 1), file_content=text
                        )

                        final_data.append(packet)
            elif file_type == ".txt":
                # loading other file types
                # st.write(eachdoc)
                text = eachdoc.read().decode("utf-8")
                # text = textract.process(bytes_data).decode("utf-8")
                packet = create_data_packet(
                    file_name, file_type, page_number=-1, file_content=text
                )
                final_data.append(packet)
        
        # st.write(final_data)
        pdf_data = pd.DataFrame.from_dict(final_data)
        # st.write(pdf_data)
        pdf_data = pdf_data.sort_values(
            by=["file_name", "page_number"]
        )  # sorting the datafram by filename and page_number
        pdf_data.reset_index(inplace=True, drop=True)

    with st.spinner('Splitting data into chunks and cleaning the text...'):
        global chunk_size
        # you can define how many words should be there in a given chunk.
        chunk_size = chunk_size_value

        pdf_data["content"] = pdf_data["content"].apply(
        lambda x: re.sub("[^A-Za-z0-9]+", " ", x)
                        )

        # Apply the chunk splitting logic here on each row of content in dataframe.
        pdf_data["chunks"] = pdf_data["content"].apply(split_text)
        # Now, each row in 'chunks' contains list of all chunks and hence we need to explode them into individual rows.
        pdf_data = pdf_data.explode("chunks")

        # Sort and reset index
        pdf_data = pdf_data.sort_values(by=["file_name", "page_number"])
        pdf_data.reset_index(inplace=True, drop=True)

    with st.spinner('Building vectors of the chunk..beep boop..taking time....'):
        if sample:
            pdf_data_sample = pdf_data.sample(sample_size)
        else:
            pdf_data_sample = pdf_data.copy()
        
        pdf_data_sample["embedding"] = pdf_data_sample["content"].apply(
        lambda x: embedding_model_with_backoff([x])
        )
        pdf_data_sample["embedding"] = pdf_data_sample.embedding.apply(np.array)
        
    st.write("Vectore Store of your documents is created.....")
    return pdf_data_sample

def get_dot_product(row):
    return np.dot(row, query_vector)

def get_context_from_question(question, vector_store, sort_index_value=2):
    global query_vector
    query_vector = np.array(embedding_model_with_backoff([question]))
    top_matched = (
        vector_store["embedding"]
        .apply(get_dot_product)
        .sort_values(ascending=False)[:sort_index_value]
        .index
    )
    top_matched_df = vector_store[vector_store.index.isin(top_matched)][
        ["file_name", "page_number", "content","chunks"]
    ]
    context = "\n".join(
        vector_store[vector_store.index.isin(top_matched)]["chunks"].values
    )
    source = f"""filenames: {",".join(top_matched_df['file_name'].value_counts().index) },
              pages: {top_matched_df['page_number'].unique()}
              """
    return context, top_matched_df,source


