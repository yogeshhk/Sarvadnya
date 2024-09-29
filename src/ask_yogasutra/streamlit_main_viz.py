import streamlit as st
import networkx as nx
import streamlit.components.v1 as components
import json
import tempfile
from graph_builder import GraphBuilder
import requests
from streamlit_agraph import agraph, Node, Edge, Config
from rdflib import URIRef, Literal
from pyvis.network import Network
import graphviz

# Set the page configuration
st.set_page_config(page_title="Graph Application", layout="wide")


# Convert NetworkX graph to agraph nodes and edges
def convert_networkx_to_agraph(nx_graph):
    nodes = [Node(id=node, label=node) for node in nx_graph.nodes()]
    edges = [Edge(source=edge[0], target=edge[1]) for edge in nx_graph.edges()]
    return nodes, edges


# Convert RDF graph to agraph nodes and edges
def convert_rdf_to_agraph(rdf_graph, namespace):
    nodes = []
    edges = []

    for s, p, o in rdf_graph:
        # Add nodes
        if isinstance(s, URIRef):
            node_id = str(s).replace(str(namespace), '')
            if not any(node.id == node_id for node in nodes):
                nodes.append(Node(id=node_id, label=node_id))

        # Add edges
        if p == namespace['connected_to']:
            source = str(s).replace(str(namespace), '')
            if isinstance(o, URIRef):
                target = str(o).replace(str(namespace), '')
            elif isinstance(o, Literal):
                target = str(o)
            else:
                continue  # Skip if o is neither URIRef nor Literal
            edges.append(Edge(source=source, target=target))

    return nodes, edges


def left_sidebar_ui():
    # Import functionality
    # st.header("Import")
    st.markdown("### Import")

    uploaded_file = st.file_uploader("Choose a graph JSON file", type="json", key="file_uploader")
    # CHANGED: Update session state only if a new file is uploaded
    if uploaded_file is not None and st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file  # it is not file name but full file_uploader obj
        # st.experimental_rerun()

    # Export functionality
    # st.header("Export Graph")
    st.markdown("### Export Graph")
    if st.button("Export Graph"):
        nx_graph = st.session_state.graph_builder.export_to_networkx()
        st.download_button(
            label="Download NetworkX Graph",
            data=json.dumps(nx.node_link_data(nx_graph)),
            file_name="graph_data.json",
            mime="application/json"
        )


def graph_visualization_by_graphviz():
    # # net = st.session_state.graph_builder.visualize_by_pyvis()
    # dot = st.session_state.graph_builder.save_pic_with_graphviz()
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
    #     # net.save_graph(tmpfile.name)
    #     svg_content = dot.pipe(format='svg').decode('utf-8')  # Generate SVG content
    #     with open(tmpfile.name, 'r', encoding='utf-8') as f:
    #         # Wrap the SVG content in a <div> with CSS for scaling
    #         html_content = f"""
    #         <div style="width:100%; height:600px; overflow:auto; border:1px solid black;">
    #             <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center;">
    #                 {svg_content}
    #             </div>
    #         </div>
    #         """
    #         # Save the HTML content to the temp file
    #         tmpfile.write(html_content.encode('utf-8'))
    #         components.html(f.read(), height=600)
    dot = st.session_state.graph_builder.save_pic_with_graphviz()
    svg_content = dot.pipe(format='svg').decode('utf-8')
    st.graphviz_chart(dot)


def graph_visualization_by_agraph():
    if 'graph_builder' in st.session_state and st.session_state.graph_builder:
        rdf_graph = st.session_state.graph_builder.get_rdf_graph()
        print(f"rdf graph {rdf_graph}")
        namespace = st.session_state.graph_builder.get_namespace()
        nodes, edges = convert_rdf_to_agraph(rdf_graph, namespace)
        print(f"rdf nodes {nodes} rdf edges {edges}")

        config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)

        return_value = agraph(nodes=nodes, edges=edges, config=config)

        if return_value:
            st.session_state.selected_element = return_value


def graph_visualization_by_pyvis():
    if 'graph_builder' in st.session_state and st.session_state.graph_builder:
        nx_graph = st.session_state.graph_builder.export_to_networkx()
        net = Network(notebook=True, width="100%", height="600px")
        net.from_nx(nx_graph)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            net.save_graph(tmpfile.name)
            with open(tmpfile.name, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600)


def middle_ui():
    st.title("Graph Application")
    if st.session_state.uploaded_file is not None:
        try:

            # the file pointer being at the end of the file after the first read. When you try to read it again,
            # there's no data left to read. To resolve this, you need to reset the file pointer to the beginning
            # before reading it again.
            # Reset the file pointer to the beginning
            st.session_state.uploaded_file.seek(0)

            graph_data = json.load(st.session_state.uploaded_file)

            # After reading, reset the file pointer again for potential future reads
            st.session_state.uploaded_file.seek(0)

            nodes, edges = st.session_state.graph_builder.import_data(graph_data)
            print(f"Imported {nodes} nodes, {edges} edges")
            visualization_method = st.selectbox(
                "Select visualization method",
                ["Graphviz", "Agraph", "Pyvis"]
            )

            if visualization_method == "Graphviz":
                graph_visualization_by_graphviz()
            elif visualization_method == "Agraph":
                graph_visualization_by_agraph()
            else:  # Pyvis
                graph_visualization_by_pyvis()

        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON file.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                st.error("Error 403: Access Forbidden. You don't have permission to access this file.")
            else:
                st.error(f"HTTP Error: {str(e)}")
        except Exception as e:
            st.error(f"Failed to import graph: {str(e)}")
        else:
            st.write("Upload a graph JSON file to view or modify the graph.")
    else:
        st.write("Upload a graph JSON file on left.")


def right_sidebar_ui():
    st.markdown("### SPARQL Query")
    query = st.text_area("Enter SPARQL Query")
    if st.button("Execute Query"):
        results = st.session_state.graph_builder.sparql_query(query)
        st.write(results.serialize(format="json"))

    st.markdown("### Information Panel")
    if 'selected_element' in st.session_state and st.session_state.selected_element:
        element_id = st.session_state.selected_element
        rdf_graph = st.session_state.graph_builder.get_rdf_graph()
        namespace = st.session_state.graph_builder.get_namespace()

        full_uri = namespace[element_id]

        if (full_uri, None, None) in rdf_graph:
            properties = st.session_state.graph_builder.get_node_properties(element_id)
            st.subheader(f"Node: {element_id}")
        elif any((full_uri, namespace['connected_to'], None) in rdf_graph):
            outgoing_edges = list(rdf_graph.objects(full_uri, namespace['connected_to']))
            if outgoing_edges:
                target = str(outgoing_edges[0]).replace(str(namespace), '')
                properties = st.session_state.graph_builder.get_edge_properties(element_id, target)
                st.subheader(f"Edge: {element_id} -> {target}")
            else:
                st.write("No outgoing edges found for the selected element.")
                return
        else:
            st.write("No information found for the selected element.")
            return

        for key, value in properties.items():
            st.text(f"{key}: {value}")
    else:
        st.write("Click on a node or edge to view its properties.")


def main_ui():
    # Initialize session state
    if 'graph_obj' not in st.session_state:
        st.session_state.graph_builder = GraphBuilder()
    if 'selected_element' not in st.session_state:
        st.session_state.selected_element = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    # Create three columns to simulate left sidebar, main content, and right sidebar
    left_sidebar, main_content, right_sidebar = st.columns([2, 4, 2])

    with left_sidebar:
        left_sidebar_ui()

    with main_content:
        middle_ui()

    with right_sidebar:
        right_sidebar_ui()
    #
    # # JavaScript callback for node/edge selection
    # st.markdown("""
    # <script>
    # function selectElement(elementType, elementId) {
    #     const data = {
    #         element_type: elementType,
    #         element_id: elementId
    #     };
    #     window.parent.postMessage(JSON.stringify(data), "*");
    # }
    # </script>
    # """, unsafe_allow_html=True)
    #
    # # Handle the selection message from JavaScript
    # if st.session_state.selected_element is None:
    #     st.session_state.selected_element = None
    #
    # def handle_selection(data):
    #     st.session_state.selected_element = (data['element_type'], data['element_id'])
    #     # st.experimental_rerun()
    #
    # st.markdown("""
    # <script>
    # window.addEventListener('message', function(event) {
    #     const data = JSON.parse(event.data);
    #     window.Streamlit.setComponentValue(data);
    # });
    # </script>
    # """, unsafe_allow_html=True)
    #
    # # Use a container to trigger rerun on selection
    # placeholder = st.empty()
    # selection = placeholder.text_input("Selected Element", key="selection_input", label_visibility="hidden")
    # if selection:
    #     data = json.loads(selection)
    #     st.session_state.selected_element = (data['element_type'], data['element_id'])
    #     placeholder.empty()
    #     # st.experimental_rerun()


if __name__ == "__main__":
    main_ui()
