import streamlit as st
from streamlit_agraph import agraph, Config
from backend.graph_constructor import GraphConstructor

st.set_page_config(page_title="Yoga Sutras Explorer - Graph Viewer", layout="wide")

class GraphViewer:
    def __init__(self, graph_constructor):
        self.graph_constructor = graph_constructor
        self.config = Config(
            width=800,
            height=600,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False
        )

    def _set_styles(self):
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

    def display_sidebar(self):
        st.sidebar.markdown("### Color Scheme")
        for pada, color in self.graph_constructor.NODE_COLORS.items():
            st.sidebar.markdown(
                f'<div><span class="color-swatch" style="background-color: {color};"></span>Pada {pada}</div>', 
                unsafe_allow_html=True
            )

    def display_graph(self):
        nodes, edges = self.graph_constructor.get_agraph_nodes_and_edges()
        return agraph(
            nodes=nodes,
            edges=edges,
            config=self.config
        )

    def display_node_info(self, selected_node: str):
        node_data = self.graph_constructor.node_dict.get(selected_node)
        if node_data:
            st.markdown(f"### Sutra {selected_node}")
            
            for key, value in node_data.items():
                if key != 'id':
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    if st.button(f"Edit {key.replace('_', ' ').title()}"):
                        st.session_state.editing = (selected_node, key)

            st.markdown("---")

    def edit_node_info(self, node_id: str, field: str):
        node_data = self.graph_constructor.node_dict[node_id]
        new_value = st.text_area("Edit value", value=node_data[field], height=150)
        if st.button("Save Changes"):
            self.graph_constructor.save_changes(node_id, field, new_value)
            st.success("Changes saved successfully!")
            st.session_state.editing = None
            st.rerun()

    def display_connected_nodes(self, selected_node: str):
        connected_nodes = self.graph_constructor.get_connected_nodes(selected_node)
        if connected_nodes:
            st.markdown("### Connected Sutras")
            for node in connected_nodes:
                st.markdown(f"- {node}")
        else:
            st.markdown("No connected sutras found.")

    def render(self):
        st.title("Sutras Graph Viewer")
        self._set_styles()
        self.display_sidebar()
        
        if 'editing' not in st.session_state:
            st.session_state.editing = None

        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_node = self.display_graph()
        
        with col2:
            if st.session_state.editing:
                self.edit_node_info(st.session_state.editing[0], st.session_state.editing[1])
            elif selected_node:
                self.display_node_info(selected_node)
                self.display_connected_nodes(selected_node)
            else:
                st.markdown("Click on a node to view sutra details and connections.")

def main():
    graph_constructor = GraphConstructor('data/graph.json')
    viewer = GraphViewer(graph_constructor)
    viewer.render()

if __name__ == "__main__":
    main()