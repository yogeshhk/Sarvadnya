# Quick Start Guide

Get up and running with the Floor Plan AI System in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key (required for AI features)
- 2GB free disk space
- Internet connection

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/floorplan-ai-system.git
cd floorplan-ai-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Verify Installation

```bash
# Run basic test
python -c "from floorplan_system import FloorPlanSystem; print('✓ Installation successful!')"
```

## Basic Usage

### Storing a Floor Plan

```python
from floorplan_system import FloorPlanSystem

# Initialize system
system = FloorPlanSystem()

# Create a floor plan
floor_plan = {
    "id": "apt_101",
    "name": "2BR Modern Apartment",
    "total_area": 85.0,
    "rooms": [
        {
            "id": "bedroom_1",
            "type": "bedroom",
            "area": 15.0,
            "dimensions": {"length": 5.0, "width": 3.0, "unit": "m"},
            "features": {"windows": 2, "doors": 1}
        },
        # ... more rooms
    ]
}

# Store it
plan_id = system.store_floor_plan(floor_plan, validate=False)
print(f"Stored: {plan_id}")
```

### Searching Floor Plans

```python
# Search using natural language
results = system.query("Find 2 bedroom apartments under 1000 sq ft")

for result in results:
    print(f"- {result['metadata']['name']}")
    print(f"  {result['metadata']['total_area']} sqm")
```

### Generating Floor Plans

```python
# Generate from description
new_plan = system.generate_floor_plan(
    "Create a 3 bedroom apartment with ensuite master and open kitchen"
)

print(f"Generated: {new_plan['name']}")
print(f"Rooms: {len(new_plan['rooms'])}")
```

### Modifying Floor Plans

```python
# Modify existing plan
modified = system.modify_floor_plan(
    plan_id="apt_101",
    command="Add a balcony to the living room"
)

print(f"Modified: {modified['name']}")
```

## Running Examples

The repository includes comprehensive examples:

```bash
# Run all examples
python examples/basic_usage.py

# Run specific example
python -c "from examples.basic_usage import example_2_natural_language_queries; example_2_natural_language_queries()"
```

## Common Tasks

### Import Multiple Plans

```python
plans = [
    {...},  # Floor plan 1
    {...},  # Floor plan 2
    {...},  # Floor plan 3
]

plan_ids = system.batch_import(plans)
print(f"Imported {len(plan_ids)} plans")
```

### Export Floor Plan

```python
# Export to JSON
json_data = system.export_floor_plan("plan_001", format="json")

# Save to file
system.export_floor_plan(
    "plan_001", 
    format="json", 
    output_path="output/plan.json"
)
```

### Get Statistics

```python
stats = system.stats()
print(f"Total plans: {stats['total_plans']}")
print(f"Average area: {stats['avg_area']} sqm")
print(f"Bedroom distribution: {stats['bedroom_distribution']}")
```

### Ask Questions

```python
answer = system.answer_question(
    "What apartments do you have with balconies?"
)
print(answer)
```

## Configuration Options

### Using Pinecone (Production)

```python
system = FloorPlanSystem(
    vector_store_type="pinecone",
    index_name="my-floor-plans"
)
```

### Using FAISS (Local Development)

```python
system = FloorPlanSystem(
    vector_store_type="faiss",
    index_name="floor-plans-dev"
)
```

### Custom API Key

```python
system = FloorPlanSystem(
    api_key="sk-your-openai-key"
)
```

## Troubleshooting

### ImportError: No module named 'openai'

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### "OpenAI API key not found"

**Solution**: Set API key in .env file
```bash
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

### FAISS index not found

**Solution**: FAISS will create index automatically on first use. No action needed.

### "Model not found" error

**Solution**: Update to latest OpenAI model
```python
# In .env file
OPENAI_MODEL=gpt-4-turbo-preview
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory for more use cases
2. **Read Documentation**: See full [README.md](README.md) for detailed docs
3. **API Reference**: Browse `src/` for detailed API documentation
4. **Customize**: Modify schemas in `src/schemas/` for your needs

## Quick Reference

| Task | Command |
|------|---------|
| Store plan | `system.store_floor_plan(plan)` |
| Search | `system.query("2 bedroom apartments")` |
| Generate | `system.generate_floor_plan("description")` |
| Modify | `system.modify_floor_plan(id, "command")` |
| Export | `system.export_floor_plan(id, format="json")` |
| Stats | `system.stats()` |
| List all | `system.list_floor_plans()` |

## Performance Tips

1. **Use FAISS for development** - Faster, no API costs
2. **Use Pinecone for production** - Better scalability
3. **Batch imports** - Use `batch_import()` for multiple plans
4. **Cache results** - Plans are automatically cached in memory
5. **Hybrid search** - Enable with `use_hybrid=True` for better accuracy

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/floorplan-ai-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/floorplan-ai-system/discussions)
- **Email**: your.email@example.com

## What's Next?

- [ ] Set up continuous integration
- [ ] Add more examples
- [ ] Implement SVG/DXF converters
- [ ] Create web UI
- [ ] Add building code validation
- [ ] Support multi-floor buildings

---

**Ready to build something amazing? Start coding!** 🚀
