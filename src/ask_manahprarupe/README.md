# 🧠 Ask Vichar-Chitre Chatbot

**मानसिक मॉडेल्सवर आधारित मराठी चॅटबॉट**

A sophisticated AI chatbot specialized in Mental Models, built with Retrieval Augmented Generation (RAG) and fine-tuning capabilities. The chatbot can understand and respond to questions about cognitive biases, decision-making frameworks, and thinking patterns in Marathi language.

## 🌟 Features

- **🔍 RAG-powered responses**: Uses LlamaIndex and ChromaDB for intelligent document retrieval
- **🎯 Fine-tuning support**: LoRA-based fine-tuning using Unsloth for specialized responses
- **🇮🇳 Multilingual support**: Primarily designed for Marathi with English fallback
- **🚀 High-performance**: Optimized with Groq API for fast inference
- **💻 User-friendly UI**: Clean Streamlit interface with model selection
- **📊 Vector database**: Persistent ChromaDB storage for efficient retrieval

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│   RAG System     │───▶│  Groq/Gemma API │
│   (app.py)      │    │   (rag.py)       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       
         │              ┌──────────────────┐              
         └─────────────▶│  Fine-tuner      │              
                        │  (fine_tune.py)  │              
                        └──────────────────┘              
                                 │                        
                        ┌──────────────────┐              
                        │   ChromaDB       │              
                        │   Vector Store   │              
                        └──────────────────┘              
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for fine-tuning)
- Groq API key ([Get it here](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ask-vichar-chitre-chatbot.git
   cd ask-vichar-chitre-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Prepare your data**
   Create text files with mental models descriptions in Marathi and place them in a `data/` directory.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Sample Data Format

Create text files with mental models in this format:

```
Confirmation Bias (पुष्टीकरण पूर्वाग्रह)

व्याख्या: आपल्या आधीच्या विश्वासांना बळकटी देणारी माहिती शोधणे आणि विरोधी माहितीकडे दुर्लक्ष करणे.

उदाहरणे:
1. राजकीय मते: फक्त त्याच न्यूज चॅनेल बघणे ज्या आपल्या राजकीय पक्षाला समर्थन देतात
2. गुंतवणूक: एखाद्या कंपनीबद्दल चांगले विचार असल्यास फक्त त्या कंपनीच्या चांगल्या बातम्या वाचणे

टाळण्याचे मार्ग:
- विरोधी मतांना देखील महत्त्व द्या
- विविध स्रोतांकडून माहिती घ्या
```

## 📚 Usage

### Basic RAG Chatbot

```python
from rag import RAGChatbot

# Initialize chatbot
chatbot = RAGChatbot(
    data_directory="./data",
    groq_api_key="your_api_key"
)

# Ask questions in Marathi
response = chatbot.get_response("Sunk cost fallacy या mental model ला मराठीत काय म्हणातात?")
print(response)
```

### Fine-tuning

```python
from fine_tune import FineTuner

# Initialize fine-tuner
fine_tuner = FineTuner(data_directory="./data")

# Prepare data and fine-tune
fine_tuner.prepare_training_data()
fine_tuner.fine_tune_model(output_dir="./fine_tuned_model")

# Generate responses with fine-tuned model
response = fine_tuner.generate_response("Anchoring bias बद्दल सांगा")
```

## 🎯 Example Questions

- `Sunk cost fallacy या mental model ला मराठीत काय म्हणातात आणि त्याचे उदाहरण द्या`
- `Confirmation bias बद्दल मराठीत सांगा`
- `Decision making मध्ये कोणते mental models वापरावे?`
- `Anchoring bias म्हणजे काय आणि ते कसे टाळावे?`

## 🔧 Configuration

### Model Settings

- **Base Model**: `google/gemma-7b-it`
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Store**: ChromaDB with persistent storage
- **Fine-tuning**: LoRA with rank 16, alpha 16

### Customization

You can modify these settings in the respective classes:

```python
# In rag.py
chatbot = RAGChatbot(
    model_name="google/gemma-7b-it",  # Change model
    embedding_model="your-embedding-model"  # Change embeddings
)

# In fine_tune.py
fine_tuner = FineTuner(
    model_name="google/gemma-7b-it",  # Base model for fine-tuning
    max_seq_length=2048  # Adjust sequence length
)
```

## 🧪 Testing

Each module includes comprehensive tests:

```bash
# Test RAG functionality
python rag_module.py

# Test fine-tuning (requires GPU)
python finetune_module.py

# Run the full application
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
ask-vichar-chitre-chatbot/
├── streamlit_app.py    # Streamlit UI application
├── rag_module.py       # RAG system implementation
├── finetune_module.py  # Fine-tuning functionality
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .env               # Environment variables (create this)
├── data/              # Your mental models data (create this)
├─


## References
- [MarathiNLP: l3cube-pune: ](https://github.com/l3cube-pune/MarathiNLP), datasets, models etc