
import plotly.graph_objects as go
import json

# Data
data = {
    "representations": [
        {
            "name": "IFC (Industry Foundation Classes)",
            "semantic_richness": 5,
            "query_capability": 4,
            "generation_support": 4,
            "bim_integration": 5,
            "standards_compliance": 5
        },
        {
            "name": "Graph-based (Node-Edge Structure)",
            "semantic_richness": 4,
            "query_capability": 5,
            "generation_support": 5,
            "bim_integration": 3,
            "standards_compliance": 3
        },
        {
            "name": "JSON Schema + Metadata",
            "semantic_richness": 4,
            "query_capability": 4,
            "generation_support": 4,
            "bim_integration": 2,
            "standards_compliance": 2
        },
        {
            "name": "DXF/SVG (Geometric Only)",
            "semantic_richness": 1,
            "query_capability": 1,
            "generation_support": 2,
            "bim_integration": 4,
            "standards_compliance": 4
        }
    ]
}

# Extract data for heatmap
approaches = [rep["name"] for rep in data["representations"]]
dimensions = ["Semantic Rich.", "Query Capable.", "Gen. Support", "BIM Integrat.", "Standards Comp."]

# Create matrix for heatmap
z_values = []
for rep in data["representations"]:
    z_values.append([
        rep["semantic_richness"],
        rep["query_capability"],
        rep["generation_support"],
        rep["bim_integration"],
        rep["standards_compliance"]
    ])

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=z_values,
    x=dimensions,
    y=approaches,
    colorscale=[
        [0, '#FFCDD2'],      # Light red for low scores
        [0.2, '#FFEB8A'],    # Light yellow
        [0.4, '#B3E5EC'],    # Light cyan
        [0.6, '#A5D6A7'],    # Light green
        [1, '#2E8B57']       # Sea green for high scores
    ],
    text=z_values,
    texttemplate="%{text}",
    textfont={"size": 16},
    hovertemplate='<b>%{y}</b><br>%{x}: %{z}/5<extra></extra>',
    colorbar=dict(
        title="Score",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["1", "2", "3", "4", "5"]
    )
))

# Update layout
fig.update_layout(
    title={
        "text": "Floor Plan Representation Approaches Comparison (1-5 Scale)<br><span style='font-size: 18px; font-weight: normal;'>IFC excels in standards; Graph-based leads in flexibility</span>"
    },
    xaxis_title="Evaluation Dimensions",
    yaxis_title="Approach Type"
)

# Save as PNG and SVG
fig.write_image("floor_plan_comparison.png")
fig.write_image("floor_plan_comparison.svg", format="svg")
