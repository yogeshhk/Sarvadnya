# AI-Powered Floor Plan Management System

A comprehensive system for semantic storage, retrieval, and AI-powered manipulation of architectural floor plans using RAG (Retrieval-Augmented Generation) and natural language interfaces.

## The Problem Statement

### Current Challenges in Architectural Floor Plan Management

**1. Inefficient Storage and Retrieval**
- Floor plans are stored as static files (PDFs, images, DXF) without semantic metadata
- Searching for specific floor plans requires manual browsing through folders
- No ability to query plans using natural language (e.g., "Find all 2-bedroom apartments under 1000 sq ft")
- Information about room relationships, adjacencies, and spatial constraints is lost in visual representations

**2. Lack of Intelligent Search**
- Architects and designers cannot search across their floor plan libraries semantically
- Retrieval is limited to filename-based searches or manual inspection
- No way to filter by architectural constraints (room count, area, features, adjacencies)
- Similar design patterns cannot be discovered automatically

**3. Tedious Manual Design Process**
- Creating floor plans from client requirements is time-consuming
- Translating natural language briefs ("3-bedroom apartment with ensuite master") into layouts requires significant manual effort
- Modifications require CAD expertise and manual redrawing
- No AI assistance for generating initial layouts or variations

**4. Disconnected Data Formats**
- Display formats (SVG, images) are separate from semantic metadata
- CAD formats (DXF, IFC) are complex and not easily queryable
- No standardized JSON schema for floor plan metadata
- Difficult to integrate with modern AI/ML tools

**5. Limited Accessibility**
- Non-technical users (clients, real estate agents) cannot easily interact with floor plan data
- Requires specialized CAD software to make even simple modifications
- No conversational interface for exploring or modifying designs

## The Solution

An AI-powered floor plan management system that:
✅ **Stores floor plans with rich semantic metadata** using a standardized JSON schema based on IFC standards
✅ **Enables natural language retrieval** using RAG (Retrieval-Augmented Generation) and vector embeddings
✅ **Provides an AI copilot** that generates and modifies floor plans from English commands
✅ **Bridges the gap** between display formats and semantic data through graph-based relationships
✅ **Democratizes floor plan interaction** by removing the need for CAD expertise

## Target Users
- **Architects & Designers**: Faster iteration, AI-assisted generation, semantic search across project libraries
- **Real Estate Developers**: Quick retrieval of floor plans matching client requirements
- **Construction Companies**: Automated code compliance checking, constraint validation
- **PropTech Startups**: Building blocks for intelligent property search and recommendation systems
- **Research Community**: Platform for advancing architectural AI, spatial reasoning, and generative design

## Expected Impact

- **10x faster** floor plan retrieval through natural language search
- **5x reduction** in time from client brief to initial layout concepts
- **Democratized access** to architectural design tools for non-CAD users
- **Standardized semantic representation** enabling AI/ML innovation in architecture
- **Open-source foundation** for next-generation architectural software

## Success Metrics

- Successfully index and retrieve 1000+ floor plans with >90% accuracy
- Generate architecturally valid floor plans from natural language in <30 seconds
- Enable non-technical users to modify floor plans through conversational commands
- Achieve community adoption with 100+ GitHub stars and 10+ contributors
- Publish research findings on architectural AI applications


**This project addresses a fundamental gap in how architectural data is stored, searched, and manipulated in the age of AI.**

## 🏗️ System Overview

This system implements a three-layer architecture for managing floor plans with AI capabilities:

1. **Storage Layer**: Multi-database approach with semantic metadata
2. **Retrieval Layer**: RAG-based natural language queries
3. **Generation Layer**: AI copilot for floor plan creation and modification

## ✨ Key Features

- **Semantic Storage**: Store floor plans with rich metadata in JSON format
- **Natural Language Retrieval**: Query floor plans using plain English
- **AI Copilot**: Generate and modify floor plans via conversational commands
- **Multi-Format Support**: Import/export DXF, SVG, JSON formats
- **Graph-Based Relationships**: Track room adjacencies and spatial relationships
- **Vector Search**: Similarity-based retrieval using embeddings
- **Code Compliance**: Validate against building codes and constraints

## 🏛️ Architecture

```
┌─────────────────────────────────────────┐
│         User Interface Layer            │
│  (Natural Language Queries & Commands)  │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│        AI Processing Layer              │
│  ┌──────────┐  ┌──────────┐            │
│  │   RAG    │  │ Copilot  │            │
│  │ Retrieval│  │ Generator│            │
│  └──────────┘  └──────────┘            │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         Storage Layer                   │
│  ┌──────────┐  ┌──────────┐            │
│  │  Vector  │  │  Graph   │            │
│  │   DB     │  │   DB     │            │
│  └──────────┘  └──────────┘            │
└─────────────────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key (for embeddings and GPT-4)
- Optional: Neo4j for graph storage
- Optional: Pinecone for production vector search

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/floorplan-ai-system.git
cd floorplan-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_key_here  # Optional
NEO4J_URI=bolt://localhost:7687           # Optional
NEO4J_USER=neo4j                          # Optional
NEO4J_PASSWORD=your_password              # Optional
```

### Basic Usage

```python
from floorplan_system import FloorPlanSystem

# Initialize system
system = FloorPlanSystem()

# Store a floor plan
floor_plan = {
    "name": "2BHK Apartment",
    "total_area": 950,
    "rooms": [
        {
            "type": "bedroom",
            "area": 180,
            "dimensions": {"length": 4.5, "width": 4.0},
            "features": {"windows": 2, "doors": 1}
        },
        {
            "type": "bedroom",
            "area": 150,
            "dimensions": {"length": 5.0, "width": 3.0},
            "features": {"windows": 1, "doors": 1}
        }
    ]
}

system.store_floor_plan(floor_plan)

# Retrieve floor plans
results = system.query("Find 2 bedroom apartments under 1000 sq ft")
print(results)

# Generate new floor plan
new_plan = system.generate_floor_plan(
    "Create a 3 bedroom apartment with open kitchen and 2 bathrooms"
)
print(new_plan)
```

## 📁 Project Structure

```
floorplan-ai-system/
├── README.md
├── requirements.txt
├── .env.example
├── setup.py
├── src/
│   ├── __init__.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── metadata_store.py      # JSON metadata storage
│   │   ├── vector_store.py        # Vector embeddings
│   │   └── graph_store.py         # Room relationships
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── rag_engine.py          # RAG implementation
│   │   └── embeddings.py          # Embedding generation
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── copilot.py             # AI copilot
│   │   ├── constraint_parser.py   # NL to constraints
│   │   └── layout_generator.py    # Floor plan generation
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── floor_plan_schema.py   # JSON schema definition
│   ├── converters/
│   │   ├── __init__.py
│   │   ├── dxf_converter.py       # DXF import/export
│   │   └── svg_converter.py       # SVG visualization
│   └── utils/
│       ├── __init__.py
│       └── validators.py          # Constraint validation
├── examples/
│   ├── basic_usage.py
│   ├── rag_retrieval.py
│   └── copilot_demo.py
└── tests/
    ├── __init__.py
    ├── test_storage.py
    ├── test_retrieval.py
    └── test_generation.py
```

## 🔍 Core Components

### 1. Semantic Storage

Floor plans are stored as JSON with rich metadata:

```json
{
  "id": "plan_001",
  "name": "2BHK Residential Unit",
  "total_area": 950,
  "rooms": [
    {
      "id": "room_001",
      "type": "bedroom",
      "area": 180,
      "dimensions": {"length": 4.5, "width": 4.0, "unit": "m"},
      "features": {"windows": 2, "doors": 1}
    }
  ],
  "adjacencies": [
    {
      "room1": "room_001",
      "room2": "room_bathroom_001",
      "type": "access"
    }
  ]
}
```

### 2. RAG Retrieval

Natural language queries are processed through:
- Query embedding generation
- Vector similarity search
- Constraint filtering
- Result ranking

### 3. AI Copilot

Converts English commands to floor plan operations:
- "Add a window to the north wall" → API commands
- "Create a 3BR apartment" → Generate new layout
- "Make the kitchen larger" → Modify existing plan

## 📊 Supported Queries

### Retrieval Queries
- "Find 2 bedroom apartments with 2 bathrooms"
- "Show apartments under 1000 sq ft"
- "Find plans with open concept living"
- "Get all studio apartments"

### Generation Commands
- "Create a 3 bedroom apartment with ensuite bathrooms"
- "Generate a floor plan with open kitchen and living room"
- "Design a 1500 sq ft house with 4 bedrooms"

### Modification Commands
- "Add a window to the master bedroom"
- "Increase kitchen size by 20%"
- "Connect the dining room to the patio"

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_retrieval.py

# Run with coverage
pytest --cov=src tests/
```

## 📖 API Reference

### FloorPlanSystem

Main interface for the system.

#### Methods

**`store_floor_plan(plan_data: dict) -> str`**
- Store a floor plan with metadata
- Returns: Floor plan ID

**`query(query_text: str, filters: dict = None) -> list`**
- Search floor plans using natural language
- Returns: List of matching floor plans

**`generate_floor_plan(description: str) -> dict`**
- Generate new floor plan from description
- Returns: Generated floor plan JSON

**`modify_floor_plan(plan_id: str, command: str) -> dict`**
- Modify existing floor plan via natural language
- Returns: Updated floor plan JSON

## 🔧 Advanced Configuration

### Vector Store Options

```python
# Use Pinecone (production)
system = FloorPlanSystem(
    vector_store="pinecone",
    pinecone_index="floor-plans"
)

# Use FAISS (local development)
system = FloorPlanSystem(
    vector_store="faiss",
    faiss_index_path="./indexes/floor_plans.index"
)
```

### Graph Database Integration

```python
# Enable Neo4j for relationship queries
system = FloorPlanSystem(
    use_graph_db=True,
    neo4j_uri="bolt://localhost:7687"
)

# Query spatial relationships
adjacent_rooms = system.graph_query(
    "MATCH (r1:Room)-[:ADJACENT_TO]->(r2:Room) RETURN r1, r2"
)
```

## 🎯 Roadmap

- [x] Core storage and retrieval
- [x] RAG-based search
- [x] Basic copilot functionality
- [ ] DXF import/export
- [ ] SVG visualization
- [ ] Building code validation
- [ ] Multi-user collaboration
- [ ] Web UI interface
- [ ] Mobile app support
- [ ] Integration with Revit/AutoCAD

## 📚 Research Foundation

This implementation is based on:
- IFC (Industry Foundation Classes) standards
- House-GAN++ and HouseDiffusion research
- Tell2Design dataset approaches
- Graph Neural Networks for spatial reasoning
- RAG patterns for architectural queries

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research papers on floor plan generation
- IFC and BuildingSMART standards
- Open-source GNN libraries
- Anthropic Claude for AI capabilities

## 📞 Contact

- Project Link: [https://github.com/yourusername/floorplan-ai-system](https://github.com/yourusername/floorplan-ai-system)
- Issues: [https://github.com/yourusername/floorplan-ai-system/issues](https://github.com/yourusername/floorplan-ai-system/issues)

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@software{floorplan_ai_system,
  title={AI-Powered Floor Plan Management System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/floorplan-ai-system}
}
```
