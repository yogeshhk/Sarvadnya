import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorGenerator:
    def __init__(self, model_dir, size=100):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_file_path = os.path.join(self.model_dir, 'tfidf.pkl')
        self.vectorizer = None
        self._load_or_init_vectorizer()

    def _load_or_init_vectorizer(self):
        if os.path.exists(self.model_file_path):
            with open(self.model_file_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        else:
            self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')

    def vectorize(self, clean_questions):
        if not os.path.exists(self.model_file_path):
            self.vectorizer.fit(clean_questions)
            with open(self.model_file_path, "wb") as f:
                pickle.dump(self.vectorizer, f)

        try:
            transformed_X_csr = self.vectorizer.transform(clean_questions)
            return transformed_X_csr.toarray()
        except Exception as e:
            print("TFIDF vectorization error:", e)
            return np.zeros((len(clean_questions), 100))  # fallback shape

    def query(self, clean_usr_msg):
        try:
            vec = self.vectorizer.transform([clean_usr_msg])
            return vec.toarray()
        except Exception as e:
            print("TFIDF query error:", e)
            return np.zeros((1, 100))  # fallback shape
