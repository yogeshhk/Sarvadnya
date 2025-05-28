# RAG on Graph

## Prompt
```
You are an expert in llamaindex coding. Need to build chatbot application PoC using streamlit ui, open source models from groq and llamaindex. Here is the problem for which code needs to be written using the RAG(retrieval augmented generation) approach.

Create graph Rag class and use following approaches to retrieve relevant nodes information 
- Add importing data and indexing phase: Create dummy graph by segmenting document or data from xls, with nodes and edges properties  Store it in networkx, with text is in the nodes in vector embedding format, also in chroma db but referred in nodes.
- convert English query to embedding, find matching from db and also neighbors from graph, after retrieval use prompt to decide best information to add as context 
- convert English query to graph query language and then receive relevant nodes, further steps as above

Use llamaindex, groq and huggingface for models, streamlit for ui. Ui file app.py should be separate and should just call functions from GraphRAG class in graphrag.py file. The graphrag.py file should have tests in __main__ section to evaluate all the functions, independent of ui.  Give only code no reasoning.
```

# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- Groq API key (get from https://console.groq.com/)

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Get your Groq API key from https://console.groq.com/
2. Replace `"your_groq_api_key_here"` in `graphrag.py` line 196 with your actual API key for testing

## Running the Application

### Streamlit UI
```bash
streamlit run app.py
```

### Testing GraphRAG Functions
```bash
python graphrag.py
```

## Usage

1. **Start the Streamlit app**: Run `streamlit run app.py`
2. **Enter API Key**: Input your Groq API key in the sidebar
3. **Upload Data**: Upload an Excel file with your data
4. **Ask Questions**: Start chatting with your data using natural language

## File Structure

- `graphrag.py` - Main GraphRAG implementation with all core functions
- `app.py` - Streamlit UI that calls GraphRAG functions
- `requirements.txt` - Python dependencies
- `sample_data.xlsx` - Generated during testing (contains sample product data)
- `chroma_db/` - ChromaDB storage directory (created automatically)

## Features

### GraphRAG Class Methods:
- `load_and_segment_data()` - Load Excel data and create segments
- `create_dummy_graph()` - Build NetworkX graph with nodes and edges
- `store_in_chroma()` - Store embeddings in ChromaDB
- `retrieve_by_embedding_similarity()` - Vector similarity search
- `get_graph_neighbors()` - Graph traversal for neighbors
- `query_to_graph_query()` - Convert natural language to graph queries
- `retrieve_by_graph_query()` - Entity-based graph retrieval
- `hybrid_retrieve()` - Combined retrieval approach
- `answer_query()` - Main query answering function

### UI Features:
- File upload for Excel data
- Real-time chat interface
- Query history with detailed results
- Sample questions
- Graph statistics display

## Testing

The `graphrag.py` file includes comprehensive tests in the `__main__` section that:
- Creates sample Excel data
- Tests all major functions
- Validates graph creation and retrieval
- Runs sample queries

## Troubleshooting

1. **API Key Issues**: Ensure your Groq API key is valid and has sufficient credits
2. **File Upload**: Make sure Excel files have proper column headers
3. **Dependencies**: Run `pip install -r requirements.txt` to install all required packages
4. **ChromaDB**: The database is created automatically in `./chroma_db/` directory