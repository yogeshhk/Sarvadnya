# # Ref: https://github.com/hanxiao/bert-as-service#building-a-qa-semantic-search-engine-in-3-minutes
# # need to have bert server running in a separate window
# # bert-serving-start -num_worker=1 -model_dir=D:/Yogesh/Education/DataScience/Datasets/uncased_L-12_H-768_A-12 -max_seq_len NONE

# from bert_serving.client import BertClient
# import numpy as np

# ## Todo : this is still working on pretrained model, not corpus specific
# class BertGenerator:
    
#     def __init__(self, model_dir, size=300):
#         self.pretrained_model_dir = model_dir + "uncased_L-12_H-768_A-12/"
#         self.model_dir = model_dir
#         self.bc = BertClient(port=5555, port_out=5556)

#     def vectorize(self, clean_questions):
#         return self.bc.encode(clean_questions)
        
#     def query(self, clean_usr_msg):
#         t_usr_array= None
#         try:
#             t_usr_array = self.bc.encode([clean_usr_msg])[0]
#         except Exception as e:
#             print(e)
#             return "Could not follow your question [" + t_usr_array + "], Try again"
            
#         return np.array([t_usr_array])
from sentence_transformers import SentenceTransformer
import numpy as np

class BertGenerator:
    def __init__(self):
        # Load a fast, compact model (you can change it later if needed)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def vectorize(self, clean_questions):
        try:
            return self.model.encode(clean_questions)
        except Exception as e:
            print("SentenceTransformer vectorize error:", e)
            return np.zeros((len(clean_questions), 384))

    def query(self, clean_usr_msg):
        try:
            emb = self.model.encode([clean_usr_msg])[0]
            return np.array([emb])
        except Exception as e:
            print("SentenceTransformer query error:", e)
            return np.zeros((1, 384))
#sentence transformer is used since bertgenerator is depricated
