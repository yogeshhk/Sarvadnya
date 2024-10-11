import streamlit as st
import networkx as nx
import json
from graph_builder import GraphBuilder
from streamlit_agraph import agraph, Node, Edge, Config
from rdflib import URIRef, Literal

st.set_page_config(page_title="Sutra Graph Viewer", layout="wide")

# Color scheme for padas
NODE_COLORS = {
    'I': "#D4A5A5",
    'II': "#9D7E79",
    'III': "#614051",
    'IV': "#A26769"
}

def get_node_color(node_id):
    pada = node_id.split('.')[0]
    return NODE_COLORS.get(pada, "#614051")  # Default color if pada not found

def convert_rdf_to_agraph(rdf_graph, namespace):
    nodes = []
    edges = []

    for s, p, o in rdf_graph:
        if isinstance(s, URIRef):
            node_id = str(s).replace(str(namespace), '')
            if not any(node.id == node_id for node in nodes):
                color = get_node_color(node_id)
                nodes.append(Node(id=node_id, label=node_id, color=color))

        if p == namespace['connected_to']:
            source = str(s).replace(str(namespace), '')
            if isinstance(o, URIRef):
                target = str(o).replace(str(namespace), '')
            elif isinstance(o, Literal):
                target = str(o)
            else:
                continue
            edges.append(Edge(source=source, target=target))

    return nodes, edges

def left_sidebar_ui():
    st.sidebar.markdown("### Import")
    use_default = st.sidebar.checkbox("Use default data/graph.json", value=True)
    
    if use_default:
        if 'graph_builder' not in st.session_state or st.session_state.graph_builder.json_file != 'data/graph.json':
            st.session_state.graph_builder = GraphBuilder('data/graph.json')
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a graph JSON file", type="json", key="file_uploader")
        if uploaded_file is not None:
            graph_data = json.load(uploaded_file)
            st.session_state.graph_builder = GraphBuilder()
            st.session_state.graph_builder.import_data(graph_data)

    st.sidebar.markdown("### Export Graph")
    if st.sidebar.button("Export Graph"):
        json_data = st.session_state.graph_builder.export_to_json()
        st.sidebar.download_button(
            label="Download JSON",
            data=json_data,
            file_name="graph_data.json",
            mime="application/json"
        )

    st.sidebar.markdown("### Color Scheme")
    for pada, color in NODE_COLORS.items():
        st.sidebar.markdown(
            f'<div><span style="display:inline-block;width:20px;height:20px;background-color:{color};margin-right:10px;"></span>Pada {pada}</div>', 
            unsafe_allow_html=True
        )

def graph_visualization():
    if 'graph_builder' in st.session_state and st.session_state.graph_builder:
        rdf_graph = st.session_state.graph_builder.get_rdf_graph()
        namespace = st.session_state.graph_builder.get_namespace()
        nodes, edges = convert_rdf_to_agraph(rdf_graph, namespace)

        config = Config(width=700, height=700, directed=True, physics=True, hierarchical=False)

        return_value = agraph(nodes=nodes, edges=edges, config=config)

        if return_value:
            st.session_state.selected_element = return_value

def information_panel():
    if 'selected_element' in st.session_state and st.session_state.selected_element:
        element_id = st.session_state.selected_element
        properties = st.session_state.graph_builder.get_node_properties(element_id)
        st.subheader(f"Sutra {element_id}")
        
        for key, value in properties.items():
            if key != 'id':
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                if st.button(f"Edit {key.replace('_', ' ').title()}"):
                    st.session_state.editing = (element_id, key)

        st.markdown("---")

        st.markdown("### Connected Sutras")
        connected_nodes = st.session_state.graph_builder.get_connected_nodes(element_id)
        for node in connected_nodes:
            st.markdown(f"- {node}")

        st.markdown("### Edit Connections")
        st.subheader("Add New Connection")
        all_nodes = st.session_state.graph_builder.get_all_node_ids()
        target_node = st.selectbox("Select target node", [node for node in all_nodes if node != element_id])
        if st.button("Add Connection"):
            st.session_state.graph_builder.add_connection(element_id, target_node)
            st.success(f"Connection added between {element_id} and {target_node}")
            st.session_state.graph_builder.save_to_file()
            st.rerun()

        st.subheader("Remove Existing Connection")
        if connected_nodes:
            node_to_remove = st.selectbox("Select connection to remove", connected_nodes)
            if st.button("Remove Connection"):
                st.session_state.graph_builder.remove_connection(element_id, node_to_remove)
                st.success(f"Connection removed between {element_id} and {node_to_remove}")
                st.session_state.graph_builder.save_to_file()
                st.rerun()
        else:
            st.write("No existing connections to remove.")
    else:
        st.write("Click on a node to view sutra details and edit connections.")

def edit_node_info(node_id: str, field: str):
    node_data = st.session_state.graph_builder.get_node_properties(node_id)
    new_value = st.text_area("Edit value", value=node_data[field], height=150)
    if st.button("Save Changes"):
        st.session_state.graph_builder.save_changes(node_id, field, new_value)
        st.session_state.graph_builder.save_to_file()
        st.success("Changes saved successfully!")
        st.session_state.editing = None
        st.rerun()

def main():
    if 'graph_builder' not in st.session_state:
        st.session_state.graph_builder = GraphBuilder()
    if 'selected_element' not in st.session_state:
        st.session_state.selected_element = None
    if 'editing' not in st.session_state:
        st.session_state.editing = None

    st.title("Sutra Graph Viewer")

    left_sidebar_ui()

    col1, col2 = st.columns([3, 1])

    with col1:
        graph_visualization()

    with col2:
        information_panel()

    if st.session_state.editing:
        edit_node_info(st.session_state.editing[0], st.session_state.editing[1])

if __name__ == "__main__":
    main()