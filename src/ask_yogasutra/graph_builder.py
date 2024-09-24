import json

import graphviz
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
from rdflib import Graph as RDFGraph, Literal, URIRef, Namespace
from rdflib.plugins.sparql import prepareQuery


class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()  # MultiGraph()
        self.rdf_graph = RDFGraph()
        self.ns = Namespace("http://example.org/")
        self.rdf_graph.bind("ex", self.ns)

    # Check if edge exists
    def check_edge_exists(self, source_node_id, target_node_id):
        # Method 1: Using get_edges()
        edges = self.graph.edges()
        for edge in edges:
            if edge[0] == source_node_id and edge[1] == target_node_id:
                return True
        return False
        #
        # # Method 2: Using get_adjacency()
        # adjacency_matrix = network.get_adjacency()
        # if adjacency_matrix[source_node_id][target_node_id] > 0:
        #     return True
        # return False

    def add_node(self, node_id, properties=None):
        self.graph.add_node(node_id)
        if properties:
            for key, value in properties.items():
                self.graph.nodes[node_id][key] = value
                self.rdf_graph.add((URIRef(self.ns[node_id]), URIRef(self.ns[key]), Literal(value)))

    def add_edge(self, source, target, properties=None):
        if self.check_edge_exists(source, target):
            return

        self.graph.add_edge(source, target)
        self.rdf_graph.add((self.ns[source], self.ns['connected_to'], self.ns[target]))
        if properties:
            for key, value in properties.items():
                self.graph[source][target][key] = value
                self.rdf_graph.add((URIRef(self.ns[source]), URIRef(self.ns[key]), URIRef(self.ns[target])))

    def get_node_properties(self, node_id):
        return self.graph.nodes[node_id]
        # properties = {}
        # for _, pred, obj in self.rdf_graph.triples((self.ns[node_id], None, None)):
        #     key = pred.split(self.ns)[-1]
        #     properties[key] = obj.toPython()
        # return properties

    def get_edge_properties(self, source, target):
        return self.graph[source][target]
        # properties = {}
        # for pred, obj in self.rdf_graph.predicate_objects(self.ns[source]):
        #     if obj == self.ns[target]:
        #         key = pred.split(self.ns)[-1]
        #         properties[key] = "True"
        # return properties

    def update_node_properties(self, node_id, properties):
        self.graph.nodes[node_id].update(properties)
        node_uri = URIRef(self.ns[node_id])
        for key, value in properties.items():
            self.rdf_graph.set((node_uri, URIRef(self.ns[key]), Literal(value)))

    def update_edge_properties(self, source, target, properties):
        self.graph[source][target].update(properties)
        source_uri = URIRef(self.ns[source])
        target_uri = URIRef(self.ns[target])
        for key, value in properties.items():
            self.rdf_graph.set((source_uri, URIRef(self.ns[key]), target_uri))

    def visualize_by_pyvis(self):
        net = Network('500px', '500px', select_menu=True, filter_menu=True)
        net.from_nx(self.graph)
        net.show('nx-before.html', False)  # Display Graph
        return net

    def visualize_with_matplotlib(self):
        # Draw the NetworkX graph using matplotlib
        plt.figure(figsize=(8, 6))
        nx.draw_networkx(self.graph, with_labels=True, font_size=10)
        plt.show()

    def save_pic_with_graphviz(self):
        dot = graphviz.Digraph(format='png')
        for node in self.graph.nodes():
            dot.node(str(node))
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            dot.edge(str(source), str(target), label=str(data.get('label', '')))
        # dot.render('graph')
        return dot

    def export_to_networkx(self):
        return self.graph.copy()

    def get_rdf_graph(self):
        return self.rdf_graph

    def get_namespace(self):
        return self.ns

    def import_from_networkx(self, nx_graph):
        self.graph = nx_graph.copy()
        self.rdf_graph = RDFGraph()
        for node, data in self.graph.nodes(data=True):
            for key, value in data.items():
                self.rdf_graph.add((URIRef(self.ns[node]), URIRef(self.ns[key]), Literal(value)))
        for source, target, data in self.graph.edges(data=True):
            for key, value in data.items():
                self.rdf_graph.add((URIRef(self.ns[source]), URIRef(self.ns[key]), URIRef(self.ns[target])))

    def sparql_query(self, query):
        prepared_query = prepareQuery(query)
        results = self.rdf_graph.query(prepared_query)
        return results

    def import_data(self, data):
        # try:
        #     with open(file_path, 'r') as file:
        #         data = json.load(file)
        # except FileNotFoundError:
        #     raise FileNotFoundError(f"File not found: {file_path}")
        # except json.JSONDecodeError:
        #     raise ValueError(f"Invalid JSON in file: {file_path}")

        self.graph = nx.Graph()
        self.rdf_graph = RDFGraph()
        self.ns = Namespace("http://example.org/")
        self.rdf_graph.bind("ex", self.ns)

        # Process nodes
        for node in data['elements']['nodes']:
            node_id = node['data']['id']
            self.add_node(node_id, node['data'])

        # Process edges
        for edge in data['elements']['edges']:
            source = edge['data']['source']
            target = edge['data']['target']
            self.add_edge(source, target, edge['data'])

        # Process positions
        for node_id, pos in data['positions'].items():
            if node_id in self.graph.nodes:
                self.update_node_properties(node_id, {'x': pos['x'], 'y': pos['y']})

        return len(self.graph.nodes), len(self.graph.edges)

    def import_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {file_path}")
        except UnicodeDecodeError:
            # If UTF-8 fails, try with 'latin-1' encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    data = json.load(file)
            except UnicodeDecodeError:
                raise ValueError(f"Unable to decode file: {file_path}. Please ensure it's a valid text file.")

        return self.import_data(data)


def test_sythetic_data():
    test_data = {
        "elements": {
            "nodes": [
                {"data": {"id": "1", "label": "Node 1"}},
                {"data": {"id": "2", "label": "Node 2"}},
                {"data": {"id": "3", "label": "Node 3"}}
            ],
            "edges": [
                {"data": {"source": "1", "target": "2", "label": "Edge 1-2"}},
                {"data": {"source": "2", "target": "3", "label": "Edge 2-3"}}
            ]
        },
        "positions": {
            "1": {"x": 100, "y": 100},
            "2": {"x": 200, "y": 200},
            "3": {"x": 300, "y": 300}
        }
    }

    # Create a GraphBuilder instance
    graph_builder = GraphBuilder()

    # Test import_data function
    nodes_count, edges_count = graph_builder.import_data(test_data)

    print(f"Imported {nodes_count} nodes and {edges_count} edges.")

    # Test some other functions
    print("\nNode properties:")
    for node in graph_builder.graph.nodes:
        print(f"Node {node}: {graph_builder.get_node_properties(node)}")

    print("\nEdge properties:")
    for edge in graph_builder.graph.edges:
        print(f"Edge {edge}: {graph_builder.get_edge_properties(*edge)}")

    # Test SPARQL query
    query = """
    SELECT ?node ?label
    WHERE {
        ?node <http://example.org/label> ?label
    }
    """
    results = graph_builder.sparql_query(query)
    print("\nSPARQL query results:")
    for row in results:
        print(f"Node: {row.node}, Label: {row.label}")


def test_file_data():
    print("\nTesting import from file:")
    file_path = "D:/Yogesh/GitHub/Sarvadnya/src/ask_yogasutra/data/graph.json"

    # Create a GraphBuilder instance
    graph_builder = GraphBuilder()
    try:
        # Use the temporary file to test import_from_file
        file_nodes_count, file_edges_count = graph_builder.import_from_file(file_path)
        print(f"Imported from file: {file_nodes_count} nodes and {file_edges_count} edges.")

        # Verify the imported data
        print("\nVerifying imported data:")
        print("Node properties:")
        for node in graph_builder.graph.nodes:
            print(f"Node {node}: {graph_builder.get_node_properties(node)}")

        print("\nEdge properties:")
        for edge in graph_builder.graph.edges:
            print(f"Edge {edge}: {graph_builder.get_edge_properties(*edge)}")

    finally:
        # Clean up the temporary file
        # os.unlink(file_path)
        print("\nImport tests completed.")


def test_visualize_data():
    print("\nTesting Visualization from file:")
    file_path = "D:/Yogesh/GitHub/Sarvadnya/src/ask_yogasutra/data/graph.json"
    # Create a GraphBuilder instance
    graph_builder = GraphBuilder()
    try:
        # Use the temporary file to test import_from_file
        file_nodes_count, file_edges_count = graph_builder.import_from_file(file_path)
        print(f"Imported from file: {file_nodes_count} nodes and {file_edges_count} edges.")

        # Verify the imported data
        # graph_builder.visualize_by_pyvis()
        # graph_builder.visualize_with_matplotlib()
        graph_builder.save_pic_with_graphviz()
    finally:
        # Clean up the temporary file
        # os.unlink(file_path)
        print("\n Visualize Data completed.")


if __name__ == "__main__":
    # # Test data
    # test_sythetic_data()
    #
    # # Test import from file
    # test_file_data()

    # Test Visualize data
    test_visualize_data()
