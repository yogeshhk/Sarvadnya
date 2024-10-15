import streamlit as st
import json
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(layout="wide", page_title="Yoga Sutras Explorer")

st.markdown("""
<style>
    .stApp {
        background-color: #F5F5F5;
    }
    .node-info {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #614051;
        font-family: 'Serif';
    }
    .color-swatch {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 10px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Each pada has different color for easier viewing
node_colors = {
    'I': "#D4A5A5",
    'II': "#9D7E79",
    'III': "#614051",
    'IV': "#A26769"
}

# Loads json data into graph
def load_graph_data(json_data):
    nodes = []
    edges = []
    
    node_dict = {node['data']['id']: node['data'] for node in json_data['elements']['nodes']}
    
    for node_id, node_data in node_dict.items():
        pada = node_id.split('.')[0]
        nodes.append(Node(
            id=node_id,
            label=node_id,
            size=30,
            color=node_colors.get(pada, "#614051"),
            shape="dot"
        ))
    
    edge_data = [(edge['data']['source'], edge['data']['target']) for edge in json_data['elements']['edges']]
    
    for source, target in edge_data:
        edges.append(Edge(source=source, target=target, type="STRAIGHT", color="#614051", width=1))
    
    return nodes, edges, node_dict, edge_data

# Display details of selected verse
def display_node_info(selected_node, node_dict):
    if selected_node:
        node_data = node_dict.get(selected_node)
        if node_data:
            st.markdown(f"### Sutra {selected_node}")
            
            for key, value in node_data.items():
                if key != 'id':
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    if st.button(f"Edit {key.replace('_', ' ').title()}"):
                        st.session_state.editing = (selected_node, key)
            
            st.markdown("---")

# Edit details of each verse
def edit_node_info(node_id, field, node_dict, json_data):
    node_data = node_dict[node_id]
    new_value = st.text_area("Edit value", value=node_data[field], height=150)
    if st.button("Save Changes"):
        node_data[field] = new_value
        for node in json_data['elements']['nodes']:
            if node['data']['id'] == node_id:
                node['data'][field] = new_value
                break
        save_json(json_data)
        st.success("Changes saved successfully!")
        st.session_state.editing = None
        st.rerun()

# Save edited detail back to graph.json
def save_json(json_data):
    with open('data/graph.json', 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=2)

# Gives List of nodes with source node as selected verse
def display_connected_nodes(selected_node, edge_data):
    connected_nodes = [target for source, target in edge_data if source == selected_node]
    if connected_nodes:
        st.markdown("### Connected Sutras")
        for node in connected_nodes:
            st.markdown(f"- {node}")
    else:
        st.markdown("No connected sutras found.")

def main():
    st.title("Sutras Graph Viewer and Editor")
    
    if 'editing' not in st.session_state:
        st.session_state.editing = None
    
    @st.cache_data
    def load_json():
        with open('data/graph.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    
    json_data = load_json()
    
    nodes, edges, node_dict, edge_data = load_graph_data(json_data)
    
    config = Config(
        width=800,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_node = agraph(nodes=nodes, edges=edges, config=config)
    
    with col2:
        if st.session_state.editing:
            edit_node_info(st.session_state.editing[0], st.session_state.editing[1], node_dict, json_data)
        elif selected_node:
            display_node_info(selected_node, node_dict)
            display_connected_nodes(selected_node, edge_data)
        else:
            st.markdown("Click on a node to view sutra details and connections.")
    
    st.sidebar.markdown("### Color Scheme")
    for pada, color in node_colors.items():
        st.sidebar.markdown(f'<div><span class="color-swatch" style="background-color: {color};"></span>Pada {pada}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()