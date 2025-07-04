# Ask Gandharva (Indian Music Bot by mrajsingh)

Gandharvas are the member of a class of celestial beings in Indian religions. They are musicians, singers and dancers.
This is PoC for a Bot that can answer qurries related to Indian classical music,especially related to Ragas.
The future idea is to create an handy AI assistance that can help people know and learn about Indian Music and Dance or even
suggest playlist for them based on the time of the day since specific Ragas are associated with specific time of the day.

## 📂 Folder Structure

```bash
ask_gandharva/
├── chainlit_main.py       # Main entry point (Chainlit UI)
├── ingest.py              # Loads PDFs, chunks & embeds into FAISS
├── data/                  # PDF sources (e.g., eGyankosh Raga guide)
├── vectorstore/db_faiss/  # FAISS index & pickle store
├── requirements.txt       # Dependencies
```

- `ingest.py` (Preprocessing)

  - Loads all PDFs from data/
  - Splits content into chunks
  - Embeds using MiniLM
  - Saves to vectorstore/db_faiss/

- `chainlit.py` (Main Chatbot)
  - Loads vectorstore
  - Initializes llama3-8b-8192 via Groq API
  - Uses a custom prompt + RetrievalQA
  - Displays source and answer

## Run

```bash
pip install -r requirements.txt
```

> `.env` file for API-Key

```bash
#Ingest your data
pip install -r requirements.txt

#run bot
chainlit run chainlit_main.py
#or
python -m streamlit run streamlit_main_360macky.py
```

[Demo Video](./demo_video.mp4)

<!-- https://github.com/user-attachments/assets/b3f134ae-74f0-4745-951b-8e62fda8b6b7 -->

References:

1. Code base: https://github.com/yogeshhk/Sarvadnya/tree/master/src/ask_bharat
2. Data: https://egyankosh.ac.in/bitstream/123456789/47383/1/Unit-1.pdf
