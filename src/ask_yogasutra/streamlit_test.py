# import streamlit as st
# import json
#
#
# def main():
#     st.title("Streamlit File Upload Test")
#
#     # File uploader
#     uploaded_file = st.file_uploader("Choose a JSON file", type="json")
#
#     if uploaded_file is not None:
#         st.write("File uploaded successfully!")
#         st.write("File details:")
#         st.json({
#             "Filename": uploaded_file.name,
#             "FileType": uploaded_file.type,
#             "FileSize": uploaded_file.size
#         })
#
#         try:
#             # Try to read and parse the JSON content
#             data = json.load(uploaded_file)
#             st.write("JSON content:")
#             st.json(data)
#         except json.JSONDecodeError:
#             st.error("The uploaded file is not a valid JSON.")
#         except Exception as e:
#             st.error(f"An error occurred while processing the file: {str(e)}")
#     else:
#         st.write("No file uploaded yet.")
#
#     # Test session state
#     if 'upload_count' not in st.session_state:
#         st.session_state.upload_count = 0
#
#     st.write(f"Number of files uploaded in this session: {st.session_state.upload_count}")
#
#     if uploaded_file is not None:
#         st.session_state.upload_count += 1
#         st.experimental_rerun()
#
#
# if __name__ == "__main__":
#     main()

# import streamlit as st
#
# uploaded_file = st.file_uploader("Choose a file", type=["json", "txt", "csv"])
#
# if uploaded_file is not None:
#     try:
#         content = uploaded_file.read()
#         st.text("File content:")
#         st.write(content)
#     except Exception as e:
#         st.error(f"Error: {e}")


import streamlit as st
from graph_builder import GraphBuilder
import json
from rdflib import Graph as RDFGraph, Literal, URIRef, Namespace
from pyvis.network import Network
from streamlit_agraph import agraph, Node, Edge, Config
import tempfile
import streamlit.components.v1 as components


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


def graph_visualization_by_agraph(graph_builder):
    if graph_builder is not None:
        rdf_graph = graph_builder.get_rdf_graph()
        print(f"rdf graph {rdf_graph}")
        namespace = graph_builder.get_namespace()
        nodes, edges = convert_rdf_to_agraph(rdf_graph, namespace)
        print(f"rdf nodes {nodes} rdf edges {edges}")

        config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)

        return_value = agraph(nodes=nodes, edges=edges, config=config)


def graph_visualization_by_pyvis(graph_builder):
    if graph_builder is not None:
        nx_graph = graph_builder.export_to_networkx()
        net = Network(height='400px', width='50%', heading='', notebook=False)
        # net.from_nx(nx_graph)
        edges = nx_graph.edges(data=True)
        nodes = nx_graph.nodes(data=True)

        if len(edges) > 0:
            for e in edges:
                net.add_node(e[0], **nodes[e[0]])
                net.add_node(e[1], **nodes[e[1]])
                print(f"e0 {e[0]}, e1 {e[1]}")
                net.add_edge(e[0], e[1])

        # net.write_html('example.html', open_browser=True, notebook=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            net.save_graph(tmpfile.name)
            with open(tmpfile.name, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600)


def main():
    st.title("Yoga Sutra RDF Graph")

    # Load JSON data
    uploaded_file_path = "D:/Yogesh/GitHub/Sarvadnya/src/ask_yogasutra/data/graph_small.json"
    graph_builder = GraphBuilder()
    with open(uploaded_file_path, 'r', encoding='utf-8') as uploaded_file:
        graph_data = json.load(uploaded_file)
        nodes, edges = graph_builder.import_data(graph_data)
        # graph_visualization_by_agraph(graph_builder)
        graph_visualization_by_pyvis(graph_builder)


if __name__ == "__main__":
    main()
