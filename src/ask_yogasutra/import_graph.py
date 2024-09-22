import networkx as nx
from rdflib import Graph as RDFGraph, Literal, URIRef, Namespace
import json

class GraphImporter:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_json_data(self):
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            st.error(f"File not found: {self.file_path}")
        except json.JSONDecodeError:
            st.error(f"Invalid JSON in file: {self.file_path}")
        return None

    def import_data(self):
        data = self.load_json_data()
        if not data:
            return None

        nx_graph = nx.Graph()
        rdf_graph = RDFGraph()
        ns = Namespace("http://example.org/")

        # Process nodes
        for node in data['elements']['nodes']:
            node_id = node['data']['id']
            nx_graph.add_node(node_id, **node['data'])
            node_uri = ns[node_id]
            for key, value in node['data'].items():
                rdf_graph.add((node_uri, ns[key], Literal(value)))

        # Process edges
        for edge in data['elements']['edges']:
            source = edge['data']['source']
            target = edge['data']['target']
            nx_graph.add_edge(source, target, **edge['data'])
            source_uri = ns[source]
            target_uri = ns[target]
            rdf_graph.add((source_uri, ns['connectedTo'], target_uri))
            for key, value in edge['data'].items():
                if key not in ['source', 'target']:
                    rdf_graph.add((source_uri, ns[key], Literal(value)))

        # Process positions
        for node_id, pos in data['positions'].items():
            if node_id in nx_graph.nodes:
                nx_graph.nodes[node_id]['x'] = pos['x']
                nx_graph.nodes[node_id]['y'] = pos['y']

        return nx_graph, rdf_graph
