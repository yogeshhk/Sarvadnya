import streamlit as st
import networkx as nx
import streamlit.components.v1 as components
import json
import tempfile
from graph_builder import GraphBuilder
import requests  # Add this import

# Set the page configuration
st.set_page_config(page_title="Graph App Title", layout="wide")


def graph_visualization():
    # net = st.session_state.graph_builder.visualize_by_pyvis()
    dot = st.session_state.graph_builder.save_pic_with_graphviz()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        # net.save_graph(tmpfile.name)
        svg_content = dot.pipe(format='svg').decode('utf-8')  # Generate SVG content
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            # Wrap the SVG content in a <div> with CSS for scaling
            html_content = f"""
            <div style="width:100%; height:600px; overflow:auto; border:1px solid black;">
                <div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center;">
                    {svg_content}
                </div>
            </div>
            """
            # Save the HTML content to the temp file
            tmpfile.write(html_content.encode('utf-8'))
            components.html(f.read(), height=600)


def middle_ui():
    st.title("Graph Application")

    if st.session_state.uploaded_file is not None:
        try:
            graph_data = json.load(st.session_state.uploaded_file)
            nodes, edges = st.session_state.graph_builder.import_data(graph_data)
            # st.success("Graph imported successfully!")
            # st.write(f"Imported {nodes} nodes and {edges} edges")
            # 
            # # Display file name
            # st.write("Uploaded file:", st.session_state.uploaded_file.name)
            # 
            # # Graph Visualization
            # st.header("Graph Visualization")
            graph_visualization()

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
        st.write("Upload a graph JSON file to view or modify the graph.")


def right_sidebar_ui():
    # # SPARQL Query Interface
    # st.header("SPARQL Query")
    # query = st.text_area("Enter SPARQL Query")
    # if st.button("Execute Query"):
    #     results = st.session_state.graph_builder.sparql_query(query)
    #     st.write(results.serialize(format="json"))

    # Export functionality
    # st.header("Export Graph")
    if st.button("Export Graph"):
        nx_graph = st.session_state.graph_builder.export_to_networkx()
        st.download_button(
            label="Download NetworkX Graph",
            data=json.dumps(nx.node_link_data(nx_graph)),
            file_name="graph_data.json",
            mime="application/json"
        )

    # st.header("Information Panel")
    if st.session_state.selected_element:
        element_type, element_id = st.session_state.selected_element
        if element_type == 'node':
            properties = st.session_state.graph_builder.get_node_properties(element_id)
            st.subheader(f"Node: {element_id}")
        else:
            source, target = element_id
            properties = st.session_state.graph_builder.get_edge_properties(source, target)
            st.subheader(f"Edge: {source} -> {target}")

        for key, value in properties.items():
            st.text(f"{key}: {value}")
    else:
        st.write("Click on a node or edge to view its properties.")


def left_sidebar_ui():
    # Import functionality
    # st.header("Import")

    uploaded_file = st.file_uploader("Choose a graph JSON file", type="json", key="file_uploader")
    # CHANGED: Update session state only if a new file is uploaded
    if uploaded_file is not None and st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        # st.experimental_rerun()


def main_ui():
    # Initialize session state
    if 'graph_obj' not in st.session_state:
        st.session_state.graph_builder = GraphBuilder()
    if 'selected_element' not in st.session_state:
        st.session_state.selected_element = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    # Create three columns to simulate left sidebar, main content, and right sidebar
    left_sidebar, main_content, right_sidebar = st.columns([2, 3, 2])

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
