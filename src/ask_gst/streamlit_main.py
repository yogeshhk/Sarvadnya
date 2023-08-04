import streamlit as st
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain

template = """
        You are a Goods and Services Tax (GST) Expert.  Under no circumstances do you give any answer outside of GST.

        ### CONTEXT
        {context}
        ### END OF CONTEXT
        
        ### QUESTION
        {question}
        ### END OF QUESTION
        
        Answer:
        """

st.title('GST FAQ ChatBot')


def generate_response(question, context=""):
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    llm = VertexAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({'context': context, 'question': question})
    st.info(response)


with st.form('my_form'):
    text = st.text_area('Ask Question:', '... about GST')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)
