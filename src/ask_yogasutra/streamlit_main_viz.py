import streamlit as st
import networkx as nx
import json
from graph_builder import GraphBuilder
from streamlit_agraph import agraph, Node, Edge, Config
from rdflib import URIRef, Literal
import colorsys

st.set_page_config(page_title="YogaSutra Graph Viewer", layout="wide")

# Custom CSS for Yoga theme (unchanged)
st.markdown("""
    <style>
    .stApp {
        background-color: #f5e6d3;
        color: #4a4a4a;
    }
    .stButton>button {
        background-color: #c77d4f;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #faf0e6;
    }
    .stSelectbox>div>div>select {
        background-color: #faf0e6;
    }
    h1, h2, h3 {
        color: #8e6343;
    }
    .node-info {
        background-color: #faf0e6;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #c77d4f;
    }
    .scrollable-text {
        max-height: 150px;
        overflow-y: auto;
        padding: 10px;
        background-color: #faf0e6;
        border: 1px solid #c77d4f;
        border-radius: 5px;
    }
    .instructions {
        background-color: #e6f0fa;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #4f77c7;
        margin-bottom: 20px;
    }
    .color-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .color-item {
        display: flex;
        align-items: center;
    }
    .color-box {
        width: 20px;
        height: 20px;
        margin-right: 5px;
        border: 1px solid #4a4a4a;
    }
    </style>
    """, unsafe_allow_html=True)

# Updated colors to better match the theme
NODE_COLORS = {
    'I': "#e6b17a",
    'II': "#d88c51",
    'III': "#c77d4f",
    'IV': "#a85c2f"
}

def generate_colors(n):
    hsv_tuples = [(x * 1.0 / n, 0.7, 0.9) for x in range(n)]
    return ['#%02x%02x%02x' % tuple(int(x*255) for x in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_tuples]

def get_node_color(node_id, properties, highlight_tags, tag_colors):
    pada = node_id.split('.')[0]
    base_color = NODE_COLORS.get(pada, "#eadb8d")
    
    node_tags = properties.get('tags', '').split(',')
    matching_tags = set(node_tags) & set(highlight_tags)
    
    if matching_tags:
        # If there are matching tags, use the color of the first matching tag
        return tag_colors[list(matching_tags)[0]]
    
    return base_color

def convert_rdf_to_agraph(rdf_graph, namespace, highlight_tags, tag_colors):
    nodes = []
    edges = []
    positions = st.session_state.graph_builder.get_node_positions()

    for s, p, o in rdf_graph:
        if isinstance(s, URIRef):
            node_id = str(s).replace(str(namespace), '')
            if not any(node.id == node_id for node in nodes):
                properties = st.session_state.graph_builder.get_node_properties(node_id)
                color = get_node_color(node_id, properties, highlight_tags, tag_colors)
                x, y = positions.get(node_id, (None, None))
                label = f"{properties.get('title', '')}\n{node_id}\n{properties.get('Devanagari_Text', '')}"
                nodes.append(Node(id=node_id, label=label, color=color, shape="box", size=25, x=x, y=y))

        if p == namespace['connected_to']:
            source = str(s).replace(str(namespace), '')
            target = str(o).replace(str(namespace), '')
            edges.append(Edge(source=source, target=target, color="#8e6343"))

    return nodes, edges

def left_sidebar_ui(graph_builder):
    with st.sidebar:
        st.markdown("### YogaSutra Graph Settings")
        use_default = st.checkbox("Use default data/graph.json", value=True)
        
        if use_default:
            if 'graph_builder' not in st.session_state or st.session_state.graph_builder.json_file != 'data/graph.json':
                st.session_state.graph_builder = GraphBuilder('data/graph.json')
        else:
            uploaded_file = st.file_uploader("Choose a graph JSON file", type="json", key="file_uploader")
            if uploaded_file is not None:
                graph_data = json.load(uploaded_file)
                st.session_state.graph_builder = GraphBuilder()
                st.session_state.graph_builder.import_data(graph_data)

        st.markdown("### Highlight Tags")
        all_tags = [tag for tag in graph_builder.get_all_tags() if tag.strip()]  
        highlight_tags = st.multiselect("Select tags to highlight", all_tags)

        st.markdown("### Select Node Details")
        default_fields = [ 'id', 'Sanskrit_Text', 'Devanagari_Text', 'Word_for_Word_Analysis', 'Translation_Bryant', 'Vyasa_commentary']
        all_fields = [field for field in graph_builder.get_all_node_fields() if field not in ['x', 'y', 'references', 'label']]
        selected_fields = st.multiselect("Choose fields to display", all_fields, default=default_fields)
        st.session_state.selected_fields = selected_fields

        if st.button("Export JSON"):
            graph_builder.save_to_file()
            st.success("JSON exported successfully!")

        st.markdown("### Pada Color Legend")
        for pada, color in NODE_COLORS.items():
            st.markdown(f"<div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: {color}; margin-right: 10px;'></div>Pada {pada}</div>", unsafe_allow_html=True)

    return highlight_tags

def graph_visualization(graph_builder, highlight_tags):
    rdf_graph = graph_builder.get_rdf_graph()
    namespace = graph_builder.get_namespace()
    tag_colors = dict(zip(highlight_tags, generate_colors(len(highlight_tags))))
    nodes, edges = convert_rdf_to_agraph(rdf_graph, namespace, highlight_tags, tag_colors)
    
    config = Config(width="100%", 
                    height=800,
                    directed=True, 
                    physics=False,
                    hierarchical=False)
    config.node = {
        "shape": "box",
        "font": {"multi": True, "size": 12}
    }
    config.link = {"color": "#8e6343"}
    config.font = {"color": "#4a4a4a"}
    
    config.view = {
        "zoom": 1, 
        "minZoom": 0.5,
        "maxZoom": 2
    }

    return_value = agraph(nodes=nodes, edges=edges, config=config)

    if return_value:
        st.session_state.selected_element = return_value

    # Display color legend for highlighted tags
    if highlight_tags:
        st.markdown("### Highlighted Tags")
        st.markdown('<div class="color-legend">', unsafe_allow_html=True)
        for tag in highlight_tags:
            color = tag_colors[tag]
            st.markdown(f'<div class="color-item"><div class="color-box" style="background-color: {color};"></div>{tag}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def information_panel(graph_builder):
    if 'selected_element' in st.session_state and st.session_state.selected_element:
        element_id = st.session_state.selected_element
        properties = graph_builder.get_node_properties(element_id)
        st.markdown(f"### Sutra {element_id}")
        
        for key in st.session_state.selected_fields:
            if key in properties:
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                
                if key in ['id', 'Sanskrit_Text', 'Devanagari_Text']:
                    st.markdown(f'<div class="scrollable-text">{properties[key]}</div>', unsafe_allow_html=True)
                elif st.session_state.get('editing') == (element_id, key):
                    new_value = st.text_area(f"Edit {key}", value=properties[key], height=150)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save Changes", key=f"save_{key}"):
                            graph_builder.save_changes(element_id, key, new_value)
                            st.success("Changes saved successfully!")
                            st.session_state.editing = None
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_{key}"):
                            st.session_state.editing = None
                            st.rerun()
                else:
                    st.markdown(f'<div class="scrollable-text">{properties[key]}</div>', unsafe_allow_html=True)
                    if key not in ['id', 'Sanskrit_Text', 'Devanagari_Text']:
                        if st.button(f"Edit {key.replace('_', ' ').title()}", key=f"edit_{key}"):
                            st.session_state.editing = (element_id, key)
                            st.rerun()

        st.markdown("---")

        st.markdown("### Connected Sutras")
        connected_nodes = graph_builder.get_connected_nodes(element_id)
        for node in connected_nodes:
            st.markdown(f"- {node}")

        st.markdown("### Edit Connections")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Add New Connection")
            all_nodes = graph_builder.get_all_node_ids()
            target_node = st.selectbox("Select target node", [node for node in all_nodes if node != element_id])
            if st.button("Add Connection"):
                graph_builder.add_connection(element_id, target_node)
                st.success(f"Connection added between {element_id} and {target_node}")
                st.rerun()

        with col2:
            st.subheader("Remove Connection")
            if connected_nodes:
                node_to_remove = st.selectbox("Select connection to remove", connected_nodes)
                if st.button("Remove Connection"):
                    graph_builder.remove_connection(element_id, node_to_remove)
                    st.success(f"Connection removed between {element_id} and {node_to_remove}")
                    st.rerun()
            else:
                st.write("No existing connections to remove.")
    else:
        st.write("Click on a node to view sutra details and edit connections.")

def main():
    if 'graph_builder' not in st.session_state:
        st.session_state.graph_builder = GraphBuilder()
    if 'selected_element' not in st.session_state:
        st.session_state.selected_element = None
    if 'editing' not in st.session_state:
        st.session_state.editing = None
    if 'selected_fields' not in st.session_state:
        st.session_state.selected_fields = ['title', 'id', 'Sanskrit_Text', 'Devanagari_Text', 'Word_for_Word_Analysis', 'Translation_Bryant', 'Vyasa_commentary']

    st.title("YogaSutra Graph Viewer")

    with st.expander("User Guide", expanded=False):
        st.markdown("""
        - Click on a node in the graph to view sutra details.
        - Use the sidebar to customize the graph view and select which fields to display.
        - Edit sutra details using the "Edit" button next to each field (except id, Sanskrit_Text, and Devanagari_Text).
        - Changes are reflected in real-time but not saved permanently.
        - Click "Export JSON" in the sidebar to save all changes to the file.
        """)

    highlight_tags = left_sidebar_ui(st.session_state.graph_builder)

    col1, col2 = st.columns([3, 1])

    with col1:
        graph_visualization(st.session_state.graph_builder, highlight_tags)

    with col2:
        information_panel(st.session_state.graph_builder)

if __name__ == "__main__":
    main()