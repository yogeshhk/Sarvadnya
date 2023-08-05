# Building a GST FAQs App
with Streamlit, Langchain, HuggingFace and VertexAI Palm APIs


Goods and Services Tax (GST) is a complex tax system, and it can be difficult to find answers to specific questions. This Medium Story will show you how to build a simple FAQs app with Streamlit, Langchain, HuggingFace and VertexAI Palm APIs

## Demo
Here is a sneak peak of my GST Query Bot, designed to tackle all your GST-related questions effortlessly.
<a href="{https://www.youtube.com/watch?v=K2EmEzVtOZ0}" title="GST FAQ - YouTube"><img src="{https://www.youtube.com/watch?v=K2EmEzVtOZ0}" alt="GST FAQ - YouTube" /></a>

[LinkedIn Post](https://www.linkedin.com/posts/yogeshkulkarni_gst-bot-huggingface-activity-7093395645473972224-O3Y8/)


## Code explanation
- App uses streamlit, a Python framework used for building interactive web applications with minimal code. It allows developers to create user interfaces and handle user input easily. In this code, Streamlit is used to create the frontend of the GST FAQs web application, including the form for asking questions and displaying the answers.
- Imports the VertexAI, PromptTemplate, and LLMChain classes from the langchain library. These classes are used to build question-answering systems. 
- VertexAI is a library within LangChain that provides language model services. In this code, it is used to create an instance of the VertexAI language model, which is used in the RetrievalQA model for answering questions.
- LLMChain is a class within LangChain that represents a language model chain. It is used to create an instance of the RetrievalQA model, which combines a language model with a retrieval model for question answering.
- FAISS is a library within LangChain that provides efficient similarity search and clustering of embeddings. In this code, it is used to build a retrieval model using the embeddings generated from the loaded documents.
- HuggingFaceHubEmbeddings is a class within LangChain that is used to generate embeddings using pre-trained models from the Hugging Face model hub. In this code, it is used to generate embeddings for the loaded documents.
- RetrievalQA is a class within LangChain that represents a retrieval-based question answering model. It combines a language model with a retrieval model to find the most relevant answer to a given question. In this code, it is used to create an instance of the RetrievalQA model using the language model from VertexAI and the retrieval model built with FAISS.
- Also imports the CSVLoader, UnstructuredHTMLLoader, and PyPDFLoader classes from the langchain library. These classes are used to load data from CSV files, HTML files, and PDF files.
- Defines a template that is used to generate answers to questions. The template includes a prompt that tells the language model that the answer should be about GST.
- The build_QnA_db() function is responsible for building the question and answer (QnA) database used in the GST FAQs web application. Here is a step-by-step explanation of its functioning:
  - Loading data from CSV file: The function starts by using the CSVLoader class from the LangChain library to load data from a CSV file. The file path is specified as ./data/nlp_faq_engine_faqs.csv. This CSV file contains frequently asked questions (FAQs) related to GST.
  - Loading data from HTML file: Next, the function uses the UnstructuredHTMLLoader class from LangChain to load additional FAQs related to GST from an HTML file. The HTML file path is specified as "data/cbic-gst_gov_in_fgaq.html".
  - Generating embeddings: After loading the data, the function uses the HuggingFaceHubEmbeddings class from LangChain to generate embeddings for the loaded documents. These embeddings capture the semantic meaning of the text and are used for similarity search in the retrieval model.
  - Building the retrieval model: The function then uses the FAISS class from LangChain to build a retrieval model. The retrieval model is created using the embeddings generated in the previous step. FAISS provides efficient similarity search and clustering capabilities, which are essential for retrieving the most relevant answers to user questions.
  - Creating the RetrievalQA model: Finally, the function creates an instance of the RetrievalQA class from LangChain. The RetrievalQA model combines a language model with the retrieval model built using FAISS. In this case, the language model is provided by the VertexAI class from LangChain. The RetrievalQA model is responsible for running the question-answering process by finding the most relevant answer from the QnA database based on the user's input question.
  - Returning the QnA chain: The function returns the created RetrievalQA model, which represents the QnA chain. This chain is then stored in the Streamlit session state for future use.
- The build_QnA_db() function plays a crucial role in setting up the QnA database and the retrieval model used in the GST FAQs web application. It loads the FAQs from different sources, generates embeddings, and builds the retrieval model to enable accurate and relevant question answering.
- Once the chain is built we can use it multiple times without repopulating it. The following function thus takes a question as input and returns an answer from the FAQs database. The answer is generated by running the question through the RetrievalQA chain.
- The my_form form allows users to enter a question about GST. When the user submits the form, the generate_response_from_db() function is called to generate an answer.
- To run the app, you can save the code above as a Python file and then run it from the command line. For example, if you saved the code as gst_faqs.py, you would run it as follows: ```python gst_faqs.py```
- The app will then open in your browser. You can then ask questions about GST, and the app will generate answers from the FAQs database.