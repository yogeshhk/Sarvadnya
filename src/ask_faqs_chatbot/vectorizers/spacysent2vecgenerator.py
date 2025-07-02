import spacy
import numpy as np

class SpacySent2VecGenerator:
    def __init__(self, model_dir=None, size=300):
        try:
            self.nlp = spacy.load('en_core_web_md')  # use md or lg for real vectors
        except OSError:
            raise RuntimeError("Spacy model 'en_core_web_md' not found. Run: python -m spacy download en_core_web_md")

    def vectorize(self, clean_questions):
        vectors = []
        for q in clean_questions:
            vec = self.nlp(q).vector
            vectors.append(vec)
        return np.array(vectors)

    def query(self, clean_usr_msg):
        try:
            return np.array([self.nlp(clean_usr_msg).vector])
        except Exception as e:
            print("Spacy vectorization error:", e)
            return np.zeros((1, 300))  # fallback
