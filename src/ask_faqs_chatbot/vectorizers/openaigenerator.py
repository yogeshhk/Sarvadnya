# # need to OpenAI api key in to environmental variable.We recommend that you set the name of the variable to OPENAI_API_KEY
# from langchain.embeddings import OpenAIEmbeddings
# import numpy as np

# class OpenAIGenerator:
    
#     def __init__(self, model_dir, size=300):
#         self.embed = OpenAIEmbeddings()

#     def vectorize(self, clean_questions):
#         return self.embed.embed_documents(clean_questions)
        
#     def query(self, clean_usr_msg):
#         t_usr_array= None
#         try:
#             t_usr_array = self.embed.embed_query(clean_usr_msg)
#         except Exception as e:
#             print(e)
#             return "Could not follow your question [" + t_usr_array + "], Try again"
#         return np.array([t_usr_array])
import os
import numpy as np
from langchain.embeddings import OpenAIEmbeddings

class OpenAIGenerator:
    def __init__(self, model_dir=None):
        # Use Groq API key and endpoint
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable not set.")

        self.embed = OpenAIEmbeddings(
            openai_api_key=groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1",  # Groq-compatible
            model="text-embedding-ada-002"  # Works with Groq
        )

    def vectorize(self, clean_questions):
        try:
            return self.embed.embed_documents(clean_questions)
        except Exception as e:
            print("Groq vectorize error:", e)
            return np.zeros((len(clean_questions), 1536))

    def query(self, clean_usr_msg):
        try:
            emb = self.embed.embed_query(clean_usr_msg)
            return np.array([emb])
        except Exception as e:
            print("Groq query error:", e)
            return np.zeros((1, 1536))
#im using groq api key