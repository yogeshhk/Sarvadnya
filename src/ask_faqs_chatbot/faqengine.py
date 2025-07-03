import os
import nltk
import pandas as pd
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.svm import SVC

from vectorizers.factory import get_vectoriser

import faiss

class FaqEngine:
    def __init__(self, faqslist, type='tfidf'):
        self.faqslist = faqslist
        self.vector_store = None
        self.stemmer = LancasterStemmer()
        self.le = LE()
        self.classifier = None
        self.build_model(type)

    def cleanup(self, sentence):
        word_tok = nltk.word_tokenize(sentence)
        stemmed_words = [self.stemmer.stem(w) for w in word_tok]
        return ' '.join(stemmed_words)

    def build_model(self, type):
        self.vectorizer = get_vectoriser(type)
        dataframeslist = [pd.read_csv(csvfile).dropna() for csvfile in self.faqslist]
        self.data = pd.concat(dataframeslist, ignore_index=True)
        self.data['Clean_Question'] = self.data['Question'].apply(self.cleanup)

        embeddings = self.vectorizer.vectorize(self.data['Clean_Question'].tolist())
        embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32
        self.data['Question_embeddings'] = list(embeddings)
        self.questions = self.data['Question'].values

        # FAISS index
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        if index.is_trained:
            index.add(embeddings)
        self.vector_store = index

        # Classification setup
        if 'Class' not in self.data.columns:
            return

        y = self.data['Class'].values.tolist()
        if len(set(y)) < 2:
            return

        y = self.le.fit_transform(y)
        trainx, testx, trainy, testy = tts(embeddings, y, test_size=0.25, random_state=42)

        self.classifier = SVC(kernel='linear')
        self.classifier.fit(trainx, trainy)

    def query(self, usr):
        try:
            cleaned_usr = self.cleanup(usr)
            t_usr_array = self.vectorizer.query(cleaned_usr).astype('float32')  # Ensure float32

            if self.classifier:
                prediction = self.classifier.predict(t_usr_array)[0]
                class_ = self.le.inverse_transform([prediction])[0]
                questionset = self.data[self.data['Class'] == class_]
            else:
                questionset = self.data

            top_k = 1
            D, I = self.vector_store.search(t_usr_array, top_k)
            question_index = int(I[0][0])
            return self.data['Answer'][question_index]

        except Exception as e:
            print(e)
            return f"Could not follow your question [{usr}], Try again."


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    faqslist = [
    os.path.join(base_path, f)
    for f in os.listdir(base_path)
    if f.endswith(".csv")
    ]
    faqmodel = FaqEngine(faqslist, 'sentence-transformer')  # or 'groq' or your chosen tag
    response = faqmodel.query("Hi")
    print(response)
