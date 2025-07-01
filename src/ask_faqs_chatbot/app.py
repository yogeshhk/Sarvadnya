# from flask import Flask
# from flask import render_template, jsonify, request
# from faqengine import *
# import random
# from os import path

# BASE_DIR = path.dirname(__file__)

# FAQs_DATA_FOLDER = path.join(BASE_DIR, "data")

# app = Flask(__name__)
# app.secret_key = '12345'

# faqs_list = [path.join(FAQs_DATA_FOLDER, "Greetings.csv"), path.join(FAQs_DATA_FOLDER, "BankFAQs.csv")]
# # ToDo: need to debug for 'gensim', 'bert', 'openai' whereas 'spacy' (only pretrained), 'tfidf' seem ok.
# faqs_engine = FaqEngine(faqs_list, "tfidf")


# def get_response(user_message):
#     return faqs_engine.query(user_message)


# @app.route('/')
# def hello_world():
#     return render_template('home.html')


# @app.route('/chat', methods=["POST"])
# def chat():
#     try:
#         user_message = request.form["text"]
#         response_text = get_response(user_message)
#         return jsonify({"status": "success", "response": response_text})
#     except Exception as e:
#         print(e)
#         return jsonify({"status": "success", "response": "Sorry I am not trained to do that yet..."})


# if __name__ == "__main__":
#     app.run(port=8080)
from flask import Flask, render_template, jsonify, request
from faqengine import FaqEngine
import os

# Base directory for locating data
BASE_DIR = os.path.dirname(__file__)
FAQs_DATA_FOLDER = os.path.join(BASE_DIR, "data")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '12345'  # Consider securing this in production

# FAQ files to load
faqs_list = [
    os.path.join(FAQs_DATA_FOLDER, "Greetings.csv"),
    os.path.join(FAQs_DATA_FOLDER, "BankFAQs.csv")
    
]

# Choose your vectorizer type: 'tfidf', 'spacy', 'sentence-transformer', 'groq', etc.
VECTORIZER_TYPE = "tfidf"  # You can change this based on what you've implemented

# Load FAQ Engine
faqs_engine = FaqEngine(faqs_list, VECTORIZER_TYPE)


@app.route('/')
def index():
    return render_template('home.html')  


@app.route('/chat', methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]
        if not user_message.strip():
            return jsonify({"status": "fail", "response": "Please enter a valid question."})
        
        response_text = faqs_engine.query(user_message)
        return jsonify({"status": "success", "response": response_text})
    except Exception as e:
        print("Error in /chat route:", e)
        return jsonify({"status": "error", "response": "Sorry, something went wrong while processing your question."})


if __name__ == "__main__":
    app.run(port=8080, debug=True)
