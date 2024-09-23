# Ask Yogasutra

A chatbot for queries on the *Yogasutra* by Patanjali.

**Ask Yogasutra** is built to explore and understand the *Yogasutra* of Patanjali, offering immediate access to the original Sutra text, its commentaries, and multiple English translations. This application also allows users to manually annotate the Sutra-s by adding tags, references, and notes for each Sutra.

Graph RAG is the next big thing, ikgai, with Sanskrit it's Specific Knowledge.

## Features

- **Graph-based Visualization:** Visualizes the *Yogasutra* and its commentaries in a knowledge graph format.
- **Manual Annotations:** Allows users to annotate Sutras with custom tags, references, and notes.
- **Query Understanding:** Chatbot built on Retrieval-Augmented Generation (RAG) techniques for answering queries on the *Yogasutra*.
- **Knowledge Graph Structure:** Nodes represent Sutras, each having associated meanings, tags, and commentaries. Edges represent relationships such as definitions and explanations between Sutras.
- **Multiple Translations:** Access to various English translations for better comprehension.
  
## How it Works

This application builds a **Knowledge Graph** of the *Yogasutra*, where:

- **Nodes** are individual aphorisms (Sutras), with properties such as meanings, tags, and associated commentaries.
- **Edges** represent relationships between Sutras, such as definitions, explanations, and thematic connections.
  
Based on **Graph Retrieval-Augmented Generation (RAG)**, the chatbot can retrieve relevant information and provide accurate answers based on the user's query.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ask-yogasutra.git
    cd ask-yogasutra
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python app.py
    ```

# Graph Editor Implementation Plan using Streamlit

## 1. Setup and Environment

1. Set up a new Python environment
2. Install required libraries:
   - Streamlit
   - NetworkX
   - Pyvis (for graph visualization)
   - RDFLib (for SPARQL support)

## 2. Basic Streamlit Application Structure

1. Create a main Streamlit application file
2. Set up the basic layout with sidebar and main content area

## 3. Graph Data Structure

1. Implement a Graph class using NetworkX as the backend
2. Add methods for adding, editing, and deleting nodes and edges
3. Implement property management for nodes and edges

## 4. User Interface Components

1. Create a node creation form
2. Implement an edge creation form
3. Develop a property editor for nodes and edges
4. Design a graph visualization component using Pyvis

## 5. Graph Visualization

1. Integrate Pyvis with Streamlit for interactive graph rendering
2. Implement node and edge selection functionality
3. Display selected node/edge properties in a side panel

## 6. Graph Operations

1. Implement graph import/export functionality using NetworkX
2. Add support for basic graph operations (e.g., finding shortest path, centrality measures)

## 7. SPARQL Query Support

1. Integrate RDFLib for SPARQL query processing
2. Create a SPARQL query interface in the Streamlit app
3. Implement query result visualization

## 8. Testing and Refinement

1. Develop unit tests for core graph operations
2. Perform usability testing and gather feedback
3. Refine the user interface based on feedback

## 9. Documentation and Deployment

1. Write user documentation and usage guidelines
2. Prepare the project for open-source release (license, README, etc.)
3. Set up a deployment pipeline (e.g., using Streamlit sharing or Heroku)

## Data Sources

The project uses datasets from existing works and community contributions, including:

- **Patanjali-project YogaSutraTrees-Giacomo-De-Luca:** A graph-based visualization project for the *Yoga Sutra* and its commentaries.
  
## Project Links

- [Documentation](https://project-patanjali.gitbook.io/yoga-sutra-trees/why-the-yoga-sutra-as-a-graph)
- [Builder](https://yogasutratrees.pages.dev/)
- [GitBook](https://project-patanjali.gitbook.io/yoga-sutra-trees/)
- [Website (Beta)](https://giacomo-de-luca.github.io/YogaSutraTrees/#)
  
## Media and Lectures

- [Lecture: Giacomo De Luca - Yoga Sutra Trees - 7th ISCLS, Auroville](https://www.youtube.com/watch?v=86wcFqKNgxg)
- [Patanjali Yoga Sutras - English Yogic Gurukul Playlist](https://www.youtube.com/playlist?list=PLAV4BpXSJLOqHHfh6BNF53wfiA_bjcde2)
- [siva-sh: Spirituality meets Technology](https://siva.sh/patanjali-yoga-sutra)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributions

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## Debugging
On error "AxiosError: Request failed with status code 403"
[Fix](https://discuss.streamlit.io/t/axioserror-request-failed-with-status-code-403/38112/12):
 Create a new folder called .streamlit and create a “config.toml” file inside it. Then add this to the file:

[server]
enableXsrfProtection = false
enableCORS = false

