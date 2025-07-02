# """
#     Interfaces to core functions for Vectorisers docs functionality
# """
# import os

# from vectorizers.tfidfvectorgenerator import TfidfVectorGenerator
# from vectorizers.doc2vecgenerator import Doc2VecGenerator
# from vectorizers.spacysent2vecgenerator import SpacySent2VecGenerator
# from vectorizers.bertgenerator import BertGenerator
# from vectorizers.openaigenerator import OpenAIGenerator


# def get_vectoriser(model_name, model_dir_path=os.path.join(os.path.dirname(os.path.abspath( __file__ )),"models")):
#     vectoriser = None
#     if model_name == "gensim":
#         vectoriser = Doc2VecGenerator(model_dir_path)
#     elif model_name == "tfidf":
#         vectoriser = TfidfVectorGenerator(model_dir_path)
#     elif model_name == "spacy":
#         vectoriser = SpacySent2VecGenerator(model_dir_path)
#     elif model_name == "bert":
#         vectoriser = BertGenerator(model_dir_path)
#     elif model_name == "openai":
#         vectoriser = OpenAIGenerator(model_dir_path)

#     return vectoriser
"""
Factory function to return the appropriate Vectoriser class.
"""

import os
from vectorizers.tfidfvectorgenerator import TfidfVectorGenerator
# from vectorizers.doc2vecgenerator import Doc2VecGenerator
from vectorizers.spacysent2vecgenerator import SpacySent2VecGenerator
from vectorizers.bertgenerator import BertGenerator
from vectorizers.openaigenerator import OpenAIGenerator

def get_vectoriser(model_name, model_dir_path=None):
    if model_dir_path is None:
        model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    model_name = model_name.lower()

    # if model_name == "gensim":
    #     return Doc2VecGenerator(model_dir_path)
    if model_name == "tfidf":
        return TfidfVectorGenerator(model_dir_path)
    elif model_name == "spacy":
        return SpacySent2VecGenerator(model_dir_path)
    elif model_name == "bert":
        return BertGenerator(model_dir_path)
    elif model_name in ("openai", "groq"):  # Treat 'groq' as OpenAI-compatible
        return OpenAIGenerator(model_dir_path)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
