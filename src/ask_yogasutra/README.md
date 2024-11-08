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

3. Download the required model:
    ```bash
    # Download llama-2-7b-chat.Q4_K_M.gguf and place it in the project root
    wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
    ```

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
   streamlit run streamlit_main_viz.py
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

### Phase 2: GraphRAG Chatbot

The chatbot implementation combines graph-based retrieval with language model generation:

1. **Backend Setup (`graphrag_backend.py`)**:
   - Uses LlamaIndex for knowledge graph creation
   - Implements citation-aware query engine
   - Handles document processing and embedding

2. **Chatbot Interface (`streamlit_main_rag.py`)**:
   ```python
   streamlit run streamlit_main_rag.py
   ```
   Features:
   - Chat interface with message history
   - JSON data upload
   - Progress tracking
   - Memory usage monitoring

3. **Configuration Requirements**:
   - LlamaCPP model setup
   - Embedding model configuration
   - Graph store initialization

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
# Initialize backend
backend = GraphRAGBackend()

# Load graph data
with open('data/graph.json', 'r') as f:
    json_data = json.load(f)
backend.setup_knowledge_base(json_data)

# Query the system
response = backend.process_query("What is the definition of yoga?")
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
   # Add to streamlit_main_viz.py
   def custom_view_config():
       return Config(
           width="100%",
           height=800,
           directed=True,
           physics=False
       )
   ```

## Contributions

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) to get started.

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

### Debug Configuration

Create `.streamlit/config.toml`:
```toml
[server]
enableXsrfProtection = false
enableCORS = false
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