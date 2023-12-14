import streamlit as st
from langchain.llms import VertexAI # Need to set GCP Credentials first
# https://ai.gopubby.com/get-started-with-google-vertex-ai-api-and-langchain-integration-360262d05216
# https://python.langchain.com/docs/integrations/llms/google_vertex_ai_palm
# from langchain import PromptTemplate, LLMChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import PyPDFLoader


from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.chains import RetrievalQA

## DO NOT RUN THIS IN ANY IDE but on command line `streamlit run streamlit_main.py`

template = """
        You are a Goods and Services Tax (GST) Expert.  Give accurate answer to the following question.
        Under no circumstances do you give any answer outside of GST.
        
        ### QUESTION
        {question}
        ### END OF QUESTION
        
        Answer:
        """

st.title('GST FAQs')

#
# def generate_response(question):
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm = VertexAI()
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     response = llm_chain.run({'question': question})
#     st.info(response)


def build_QnA_db():
    loader = CSVLoader(file_path='./data/nlp_faq_engine_faqs.csv')
    docs = loader.load()

    # loader = PyPDFLoader("./data/Final-GST-FAQ-edition.pdf")
    # docs = loader.load_and_split()

    loader = UnstructuredHTMLLoader("data/cbic-gst_gov_in_fgaq.html")
    docs += loader.load()

    embeddings = HuggingFaceHubEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriver = db.as_retriever()
    llm = VertexAI() # model_name="gemini-pro", deafult=
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriver, verbose=False, chain_type="stuff")
    return chain


if "chain" not in st.session_state:
    st.session_state["chain"] = build_QnA_db()


def generate_response_from_db(question):
    chain = st.session_state["chain"]
    response = chain.run(question)
    st.info(response)


with st.form('my_form'):
    text = st.text_area('Ask Question:', '... about GST')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response_from_db(text)
