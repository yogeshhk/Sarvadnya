# Ask Yogasutra

A chatbot for queries on the *Yogasutra* by Patanjali.

**Ask Yogasutra** is built to explore and understand the *Yogasutra* of Patanjali, offering immediate access to the original Sutra text, its commentaries, and multiple English translations. This application also allows users to manually annotate the Sutra-s by adding tags, references, and notes for each Sutra.

Graph RAG is the next big thing, IKIGAI, with Sanskrit it's Specific Knowledge. Planning to incorporate GraphRAG on Togasutra-Graph to be able to answer multi-hop queries.

## Features

- **Graph-based Visualization:** Visualizes the *Yogasutra* and its commentaries in a knowledge graph format.
- **Manual Annotations:** Allows users to annotate Sutras with custom tags, references, and notes.
- **Query Understanding:** Chatbot built on Retrieval-Augmented Generation (RAG) techniques for answering queries on the *Yogasutra*.
- **Knowledge Graph Structure:** Nodes represent Sutras, each having associated meanings, tags, and commentaries. Edges represent relationships such as definitions and explanations between Sutras.
- **Multiple Translations:** Access to various English translations for better comprehension.
- **Index Persistence:** Automatically saves and loads vector store and knowledge graph indices to avoid rebuilding on each session.

## Installation and Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ask-yogasutra.git
    cd ask-yogasutra
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Get Groq API Key:
    - Sign up at https://console.groq.com/
    - Generate an API key
    - Add it to your Environment settins

<!-- 3. Download the required model:
    ```bash
    # Download llama-2-7b-chat.Q4_K_M.gguf and place it in the project root
    wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
    ``` -->

## Supported Models

This project uses Groq API with the following open-source models:
- `llama-3.3-70b-versatile` (default, most capable)
- `llama-3.1-8b-instant` (faster, good for simple queries)
- `mistral-saba-24b` (good balance)

To change the model, update `GROQ_MODEL_NAME` in the backend files.

## Implementation Guide

### Phase 1: Graph Visualization

The graph visualization component is implemented using Streamlit and includes the following key features:

1. **Graph Builder (`graph_builder.py`)**:
   - Manages the graph structure using NetworkX
   - Handles node and edge operations
   - Supports RDF integration for semantic queries
   - Provides JSON import/export functionality

2. **Visualization Interface (`streamlit_main_viz.py`)**:
   ```python
   streamlit run streamlit_main_visualization.py
   ```
   Features:
   - Interactive graph visualization using Streamlit-Agraph
   - Color-coded nodes based on Pada (chapter)
   - Tag-based highlighting
   - Node property editing
   - Connection management

3. **Customization Options**:
   - Field selection for node details
   - Tag-based filtering
   - Graph layout controls
   - Export capabilities

### Phase 2: GraphRAG Chatbot & Benchmark Testing

The chatbot implementation combines graph-based retrieval with language model generation, now enhanced with comprehensive benchmark testing capabilities:

1. **Backend Setup (`graphrag_backend.py`)**:
   - Uses LlamaIndex for knowledge graph creation
   - Implements citation-aware query engine
   - Handles document processing and embedding
   - Automatic index persistence and loading
   - Configurable ChatMode for different conversation styles

2. **Chatbot Interface (`streamlit_main_graphrag.py`)**:
   ```python
   streamlit run streamlit_main_graphrag.py
   ```
   Features:
   - Chat interface with message history
   - JSON data upload
   - Progress tracking
   - Memory usage monitoring
   - Index persistence management

3. **Benchmark Testing Suite (`test_framework.py`, `run_benchmark.py`)**:
   ```bash
   # Run single configuration
   python run_benchmark.py --config baseline_fast

   # Run all configurations with comparison
   python run_benchmark.py --all

   # List available configurations
   python run_benchmark.py --list-configs
   ```
   Features:
   - 23 comprehensive test cases across 4 categories
   - 14 configurable test scenarios
   - Multiple ChatMode testing (condense_plus_context, context, condense_question)
   - Cached index utilization options
   - Automated evaluation metrics (keyword match, sutra reference accuracy, semantic similarity)

4. **Configuration Requirements**:
   - LlamaCPP model setup
   - Embedding model configuration
   - Graph store initialization
   - ChatMode configuration for conversational testing

### Phase 3: Index Persistence

Both Linear RAG and Graph RAG backends now support automatic persistence:

1. **Automatic Persistence**:
   - Indices are automatically saved to the `models/` directory after creation
   - On subsequent runs, the system checks for existing indices
   - If data hasn't changed, the persisted index is loaded instead of rebuilding
   - Significantly reduces startup time for large datasets

2. **Persistence Directories**:
   - Linear RAG: `models/linearrag/`
   - Graph RAG: `models/graphrag/`

3. **Smart Rebuild Detection**:
   - Computes MD5 hash of input data
   - Checks if embedding model has changed
   - Automatically rebuilds if data or configuration changes
   - Manual rebuild option available in UI

4. **Force Rebuild**:
   - Use the "Force rebuild index" checkbox in the sidebar
   - Useful when you want to regenerate indices with different parameters
   - Deletes old indices and creates new ones

5. **Metadata Tracking**:
   - Each persisted index includes metadata:
     - Data hash (for change detection)
     - Embedding model name
     - LLM model name

## Usage Examples

### Graph Visualization

```python
# Initialize graph with default data
graph_builder = GraphBuilder('data/graph.json')

# Add custom node
graph_builder.add_node('1.1', {
    'Sanskrit_Text': 'अथ योगानुशासनम्',
    'Translation': 'Now, the exposition of Yoga begins'
})

# Add connection
graph_builder.add_connection('1.1', '1.2')

# Save changes
graph_builder.save_to_file()
```

### GraphRAG Queries

```python
# Initialize backend with default persistence directory
backend = GraphRAGBackend()

# Load graph data (will use persisted index if available)
with open('data/graph.json', 'r') as f:
    json_data = json.load(f)
backend.setup_knowledge_base(json_data)

# Query the system
response = backend.process_query("What is the definition of yoga?")
print(response)
```

### Using Custom Persistence Directory

```python
# Initialize with custom persistence directory
backend = GraphRAGBackend(persist_dir="my_custom_models/graphrag")

# Force rebuild even if persisted index exists
backend.setup_knowledge_base(json_data, force_rebuild=True)
```

### LinearRAG with Persistence

```python
from linearrag_backend import LinearRAGBackend

# Initialize backend
backend = LinearRAGBackend()

# Load data (automatically uses persisted index if available)
with open('data/graph.json', 'r') as f:
    json_data = json.load(f)
backend.setup_knowledge_base(json_data)

# Query
response = backend.process_query("Explain citta vritti nirodha")
print(response)
```

## Development and Extension

### Adding New Features

1. **Custom Node Properties**:
   ```python
   # Add new property to graph_builder.py
   def add_custom_property(self, node_id, property_name, value):
       self.update_node_properties(node_id, {property_name: value})
   ```

2. **New Visualization Options**:
   ```python
   # Add to streamlit_main_graphrag.py
   def custom_view_config():
       return Config(
           width="100%",
           height=800,
           directed=True,
           physics=False
       )
   ```

## Troubleshooting

### Common Issues

1. **Graph Visualization**:
   - If nodes don't appear: Check JSON format
   - If layout is cluttered: Adjust physics settings
   - If colors don't update: Clear browser cache

2. **ChatBot**:
   - Memory issues: Adjust batch size
   - Slow responses: Check GPU configuration
   - Missing model: Verify model path

3. **Index Persistence**:
   - **Index won't load**: Check `models/` directory permissions and ensure metadata.json exists
   - **Using stale data**: Use "Force rebuild index" checkbox in UI or `force_rebuild=True` in API
   - **Errors after updates**: Delete `models/` folder and rebuild, or use `python manage_indices.py clear all`
   - **Out of disk space**: Check storage with `python manage_indices.py size` and clear unused indices
   - **Storage location**: Indices stored in `models/linearrag/` and `models/graphrag/`
   - **API errors**: Ensure `persist()` is called with positional argument: `storage_context.persist(persist_dir)`

### Managing Persisted Indices

The project includes a management tool for persisted indices:

```bash
# List all persisted indices with metadata
python manage_indices.py list

# Show detailed information about indices
python manage_indices.py info linearrag
python manage_indices.py info graphrag
python manage_indices.py info all

# Check storage usage
python manage_indices.py size

# Clear indices (with confirmation)
python manage_indices.py clear linearrag
python manage_indices.py clear graphrag
python manage_indices.py clear all
```

Or manually remove directories:

```bash
# Remove all persisted indices
rm -rf models/

# Remove only Linear RAG index
rm -rf models/linearrag/

# Remove only Graph RAG index
rm -rf models/graphrag/
```

### Debug Configuration

Create `.streamlit/config.toml`:
```toml
[server]
enableXsrfProtection = false
enableCORS = false
```

## Performance Optimization

### Index Persistence Benefits

**Performance Improvements:**
- **First Run**: Building index from scratch (slow)
  - Linear RAG: ~2-5 minutes for full dataset
  - Graph RAG: ~3-7 minutes for full dataset
- **Subsequent Runs**: Loading from disk (fast) - **15-40x speedup**
  - Linear RAG: ~5-10 seconds
  - Graph RAG: ~10-15 seconds

**Storage Requirements:**
- Linear RAG index: ~50-200 MB
- Graph RAG index: ~100-300 MB
- Varies based on dataset size

**How It Works:**
1. First run builds and saves index to `models/` directory
2. Subsequent runs check data hash and load from disk if unchanged
3. System automatically rebuilds if data or model configuration changes
4. Force rebuild option available via UI checkbox or API parameter

## Testing

### Running Tests

```bash
# Run comprehensive persistence tests
python test_persistence.py

# Run example scripts
python example_persistence.py
```

The test suite verifies:
- Index creation and persistence
- Loading from persisted indices
- Query functionality
- Force rebuild functionality
- Performance improvements

### Benchmark Testing Suite

The enhanced benchmark testing framework provides comprehensive evaluation capabilities:

#### Quick Start

```bash
# Run single configuration
python run_benchmark.py --config baseline_fast

# Run all configurations with comparison report
python run_benchmark.py --all --verbose

# List available configurations
python run_benchmark.py --list-configs
```

#### Test Configurations

The framework includes 14 predefined configurations testing different aspects:

**Core Configurations:**
- `baseline_fast`: Fast model with cached indexes
- `baseline_quality`: Higher quality model with more tokens
- `query_mode`: Non-conversational mode

**ChatMode Testing:**
- `condense_plus_context`: Default conversation mode (condenses conversation + context)
- `context_chat_mode`: Direct context retrieval
- `condense_question_mode`: Standalone question generation

**Advanced Testing:**
- `tree_summarize`: Tree summarize response mode
- `refine_mode`: Iterative refinement responses
- `better_embeddings`: Enhanced semantic understanding
- `larger_chunks`: Different chunking strategies
- `full_graph`: Complete dataset testing

**Backend Comparison:**
- `linearrag_baseline`: Linear RAG for comparison
- `high_precision`: Optimal settings combination

#### Configuration Fields

Each test configuration supports:

```json
{
  "conversation_mode": true,           // Enable conversation context
  "chat_mode": "condense_plus_context", // ChatMode when conversation_mode=true
  "use_cached_index": true,            // Use persisted indexes
  "backend_type": "graphrag",          // "graphrag" or "linearrag"
  "response_mode": "compact",          // Query response mode
  "graph_file": "data/graph_small.json" // Dataset to use
}
```

#### ChatMode Options

When `conversation_mode` is enabled, different ChatMode options provide various conversation handling strategies:

- **`condense_plus_context`**: Condenses conversation history into standalone question, then retrieves context
- **`context`**: Direct context retrieval from knowledge base without conversation condensing
- **`condense_question`**: Generates standalone question from conversation for precise querying
- **`simple`**: Basic chat responses without retrieval augmentation

#### Cached Index Utilization

The `use_cached_index` option controls whether to leverage persisted indexes:

- **`true`** (default): Load existing indexes for faster testing
- **`false`**: Force index rebuild for fresh evaluation

#### Benchmark Dataset

The test suite includes 23 comprehensive test cases across 4 categories:

- **Atomic queries** (10): Basic definition and fact-checking questions
- **Multi-hop queries** (5): Questions requiring multiple reasoning steps
- **Conversational queries** (4): Follow-up questions with context
- **Complex queries** (4): Multi-part analytical questions

#### Evaluation Metrics

Three comprehensive evaluation metrics provide thorough assessment:

1. **Keyword Match Score**: Percentage of expected keywords found in responses
2. **Sutra Reference Score**: F1 score for accurate sutra citations
3. **Semantic Similarity Score**: Cosine similarity between expected and actual response embeddings

Overall pass/fail determined by average score ≥0.6 threshold.

## Management Tools

### Index Management CLI

The project includes `manage_indices.py` for managing persisted indices:

```bash
# List all indices with status and metadata
python manage_indices.py list

# Show detailed file information
python manage_indices.py info [linearrag|graphrag|all]

# Display storage usage
python manage_indices.py size

# Clear indices (with confirmation)
python manage_indices.py clear [linearrag|graphrag|all]
```

### Example Output

```
📊 Persisted Indices Status
==================================================

✓ Linear RAG
  Location: models/linearrag
  Size: 45.23 MB
  Data Hash: abc123def456...
  Embedding Model: sentence-transformers/all-MiniLM-L6-v2
  LLM Model: llama3-70b-8192

✓ Graph RAG
  Location: models/graphrag
  Size: 78.91 MB
  Data Hash: xyz789abc123...
  Embedding Model: sentence-transformers/all-MiniLM-L6-v2
  LLM Model: llama-3.1-8b-instant

==================================================
Total Storage: 124.14 MB
```

## API Reference

### LinearRAGBackend

```python
class LinearRAGBackend:
    def __init__(self, persist_dir: str = "models/linearrag"):
        """Initialize backend with optional custom persistence directory."""
        
    def setup_knowledge_base(
        self, 
        json_data: Dict[str, Any], 
        progress_callback=None,
        force_rebuild: bool = False
    ) -> bool:
        """Setup knowledge base with automatic persistence.
        
        Args:
            json_data: The JSON data containing the knowledge base
            progress_callback: Optional callback for progress updates
            force_rebuild: If True, rebuild the index even if persisted version exists
        """
        
    def process_query(self, query: str) -> str:
        """Process a query and return response with citations."""
```

### GraphRAGBackend

```python
class GraphRAGBackend:
    def __init__(self, persist_dir: str = "models/graphrag"):
        """Initialize backend with optional custom persistence directory."""
        
    def setup_knowledge_base(
        self, 
        json_data: Dict[str, Any], 
        progress_callback=None,
        force_rebuild: bool = False
    ) -> bool:
        """Setup knowledge base with automatic persistence.
        
        Args:
            json_data: The JSON data containing the knowledge base
            progress_callback: Optional callback for progress updates
            force_rebuild: If True, rebuild the index even if persisted version exists
        """
        
    def process_query(self, query: str) -> str:
        """Process a query and return response with citations."""
        
    def process_conversation(
        self, 
        query: str, 
        msgs: List[ChatMessage]
    ) -> str:
        """Process a conversational query with context."""
```

## Quick Reference

### Common Commands

```bash
# Run Streamlit apps
streamlit run streamlit_main_linearrag.py
streamlit run streamlit_main_graphrag.py

# Manage indices
python manage_indices.py list
python manage_indices.py size

# Run tests
python test_persistence.py
python example_persistence.py

# Benchmark Testing Suite
python run_benchmark.py --list-configs                    # List all configurations
python run_benchmark.py --config baseline_fast           # Run single config
python run_benchmark.py --all                           # Run all configs with comparison
python run_benchmark.py --config context_chat_mode      # Test different ChatMode
python run_benchmark.py --config no_cache_rebuild       # Force fresh index rebuild
```

### Key Features

- ✅ **Automatic Persistence**: Indices saved/loaded automatically
- ✅ **Smart Rebuild**: Detects data changes via MD5 hash
- ✅ **Force Rebuild**: UI checkbox or `force_rebuild=True` parameter
- ✅ **Management Tools**: CLI tool for index management
- ✅ **15-40x Speedup**: Fast loading on subsequent runs

### Directory Structure

```
models/                      # Persisted indices (gitignored)
├── linearrag/              # Linear RAG index
│   ├── docstore.json
│   ├── index_store.json
│   ├── vector_store.json
│   └── metadata.json
└── graphrag/               # Graph RAG index
    ├── docstore.json
    ├── index_store.json
    ├── vector_store.json
    ├── graph_store.json
    └── metadata.json
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## References

- **Patanjali-project YogaSutraTrees-Giacomo-De-Luca:** A graph-based visualization project for the *Yoga Sutra* and its commentaries.
  - [Documentation](https://project-patanjali.gitbook.io/yoga-sutra-trees/why-the-yoga-sutra-as-a-graph)
  - [Builder](https://yogasutratrees.pages.dev/)
  - [GitBook](https://project-patanjali.gitbook.io/yoga-sutra-trees/)
  - [GitHub](https://github.com/Giacomo-De-Luca/YogaSutraTrees)
  - [Website (Beta)](https://giacomo-de-luca.github.io/YogaSutraTrees/#)
  - [Lecture: Giacomo De Luca - Yoga Sutra Trees - 7th ISCLS, Auroville](https://www.youtube.com/watch?v=86wcFqKNgxg)
- [Patanjali Yoga Sutras - English Yogic Gurukul Playlist](https://www.youtube.com/playlist?list=PLAV4BpXSJLOqHHfh6BNF53wfiA_bjcde2)
- [siva-sh: Spirituality meets Technology](https://siva.sh/patanjali-yoga-sutra)
- [Graph-Code: A Multi-Language Graph-Based RAG System](https://github.com/vitali87/code-graph-rag)
