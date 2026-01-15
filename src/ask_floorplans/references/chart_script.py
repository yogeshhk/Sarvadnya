
# Create a Mermaid diagram for the three-layer floor plan management architecture

diagram_code = """
graph TB
    subgraph IL[" "]
        direction LR
        NLQ[Natural Language Query]
        AC[Architectural Copilot<br/>LLM GPT/Claude<br/>RAG Framework]
        RC[Revit/CAD Integration<br/>pyRevit API]
    end
    
    subgraph GRL[" "]
        direction LR
        FPG[Floor Plan Generator<br/>HouseDiffusion<br/>House-GAN++]
        RAG[RAG Retrieval<br/>Vector Embeddings]
        SA[Semantic Analysis<br/>Graph Neural Networks]
    end
    
    subgraph SL[" "]
        direction LR
        GDB[Graph Database<br/>Neo4j]
        VS[Vector Store<br/>Pinecone/Weaviate]
        DS[Document Store<br/>MongoDB/Firestore]
    end
    
    %% Layer labels
    IL_Label["<b>INTERACTION LAYER</b>"]
    GRL_Label["<b>GENERATION & RETRIEVAL LAYER</b>"]
    SL_Label["<b>STORAGE LAYER</b>"]
    
    %% Connect layer labels to subgraphs
    IL_Label -.-> IL
    GRL_Label -.-> GRL
    SL_Label -.-> SL
    
    %% Data flow arrows
    NLQ --> AC
    AC --> RC
    AC --> RAG
    AC --> FPG
    
    FPG --> SA
    RAG --> SA
    RAG --> AC
    FPG --> RC
    
    SA --> GDB
    RAG --> VS
    SA --> VS
    FPG --> DS
    
    GDB --> SA
    VS --> RAG
    DS --> FPG
    
    style IL fill:#B3E5EC,stroke:#1FB8CD,stroke-width:3px
    style GRL fill:#FFEB8A,stroke:#D2BA4C,stroke-width:3px
    style SL fill:#A5D6A7,stroke:#2E8B57,stroke-width:3px
"""

# Use the helper function to create and save the diagram
create_mermaid_diagram(diagram_code, 'architecture_diagram.png', 'architecture_diagram.svg')
