# AI-Powered Floor Plan Management System

A comprehensive system for semantic storage, retrieval, and AI-powered manipulation of architectural floor plans using RAG (Retrieval-Augmented Generation) and natural language interfaces.

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