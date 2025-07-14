# Wikipedia Bot

QnA bot on Wikipedia using LangChain and Palm

- Loads your API keys (like GROQ_API_KEY) from a .env file into environment variables.
- Uses **LangChain’s** **WikipediaLoader** to download summaries for selected topics.
- You get a list of Document objects.
- Wikipedia articles are long. You break them into smaller overlapping chunks so the model can handle them.
- Converts each chunk into a **vector** (numeric representation of meaning).
- Uses a free model **"llama3-70b-8192"** for this.
- Stores the chunk embeddings using **ChromaDB** so that you can later do fast semantic search.
- When a question is asked, it retrieves the **top 2 most relevant chunks**.
- Uses Groq’s hosted LLaMA3 model via OpenAI-compatible API.
- It will generate answers based on retrieved context.

### Commands:

```bash
pip install langchain langchain-community langchain-openai chromadb wikipedia huggingface-hub sentence-transformers python-dotenv
```

`.env`

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Run

```bash
python main_driver.py
```

Reference
https://python.langchain.com/docs/integrations/document_loaders/wikipedia/
