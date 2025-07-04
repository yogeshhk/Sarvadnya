# FAQs ChatBot

Getting answer automatically is magic!! its real AI (remember, the Turing Test?)

This project is a Simple Question-Answer (atomic query) based chatbot framework. Uses similarity based on different vectorizers, to find the matching question then responds with its corresponding answer.

Application Scope:

- Huge demand to take care of mundane queries
- Scales (leverage, automation, passive)
- Not much work in vernacular chatbot (serve humanity)

## How It Works

1.  **Load Data**: Load one or more `.csv` files with columns: `Question`, `Answer`, and optionally `Class`
2.  **Preprocess**: Questions are tokenized and stemmed
3.  **Vectorize**: Text is vectorized using selected backend (`TF-IDF`, `SpaCy`, etc.)
4.  **Build FAISS Index**: All vectorized questions are stored in FAISS for fast similarity matching
5.  **User Asks**: The user enters a query in the Flask frontend
6.  **Query Flow**:
    -> Predict category (if classification is enabled)
    -> Vectorize user query
    -> Search most similar question in FAISS
    -> Return the corresponding answer

##

- User inputs a question in the chatbot UI.
- Flask app (app.py) sends this to the backend.
- The FaqEngine:
  - Text Cleaning: Uses NLTK to tokenize and stem each question for consistency
  - Converts it into a vector
  - Uses FAISS to find the most similar question in the dataset
- The matching answer is returned .

## Commands:

```bash
pip install -r requirements.txt
#Also download the SpaCy model if you're using spacy vectorizer:
python -m spacy download en_core_web_sm

#Run the app
python app.py
```

[Demo Video](./demo_video.mp4)
<!-- https://github.com/user-attachments/assets/977908f3-884a-4b82-a18f-12d4893cb5e3 -->

## References

- Bhavani Ravi’s event-bot [code](https://github.com/bhavaniravi/rasa-site-bot), Youtube [Video](https://www.youtube.com/watch?v=ojuq0vBIA-g)
- Banking FAQ Bot [code](https://github.com/MrJay10/banking-faq-bot)
