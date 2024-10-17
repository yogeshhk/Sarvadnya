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


## References

The project uses datasets from existing works and community contributions, including:

- **Patanjali-project YogaSutraTrees-Giacomo-De-Luca:** A graph-based visualization project for the *Yoga Sutra* and its commentaries.

	- [Documentation](https://project-patanjali.gitbook.io/yoga-sutra-trees/why-the-yoga-sutra-as-a-graph)
	- [Builder](https://yogasutratrees.pages.dev/)
	- [GitBook](https://project-patanjali.gitbook.io/yoga-sutra-trees/)
	- [GitHub](https://github.com/Giacomo-De-Luca/YogaSutraTrees)
	- [Website (Beta)](https://giacomo-de-luca.github.io/YogaSutraTrees/#)
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

# Ask Yogasutra - Phase 1: Visualizing and Editing Verses Graph

## Overview

This project aims to create an interactive graph-based representation of Patanjali's Yogasutra using **Streamlit**. Each node represents a verse (sutra), and edges connect related verses. Users can view verse details by selecting a node, edit the graph, and save changes back to a JSON file.

## Workflow

1. **Import JSON of Verses**:
   - Load a JSON file containing verses and their relationships.

2. **Visualize the Graph**:
   - Construct an interactive graph using libraries like **Graphviz**, **Pyvis**, and **Agraph**.
   - Each node displays verse information; edges indicate direct relationships.

3. **Select and View Verse Properties**:
   - Click on a node to view detailed properties of the selected verse.

4. **Edit the Graph**:
   - Modify connections and properties directly in the graph interface.

5. **Save Changes**:
   - Updated information is saved back to the JSON file.

## Libraries Used

- **Streamlit**: For building the web application.
- **Graphviz**: For creating visual graphs.
- **Pyvis**: For interactive graph visualizations.
- **Agraph**: For additional graph handling capabilities.

# Ask Yogasutra - Phase 2: Chatbot Integration with Graph Retrieval

## Overview

In Phase 2, we aim to develop a chatbot that leverages the graph created in Phase 1 to retrieve information from the Patanjali Yogasutra verses. The chatbot will utilize Graph Retrieval-Augmented Generation (RAG) to answer users' questions based on the interconnected verses.

## Workflow

1. **Integrate the Chatbot**:
   - Implement a chatbot interface within the existing Streamlit application.

2. **Graph Retrieval**:
   - Use the graph data structure to access relevant verses based on user queries.
   - Implement Graph RAG techniques to enhance answer generation.

3. **Answering Questions**:
   - The chatbot will process user input and retrieve relevant verse information.
   - Generate responses based on the relationships and content of the verses.

4. **User Interaction**:
   - Allow users to ask questions and receive answers in a conversational format.

## Libraries Used

- **Streamlit**: For the web application interface.

