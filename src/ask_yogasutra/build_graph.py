import networkx as nx
from pyvis.network import Network
from rdflib import Graph as RDFGraph, Literal, URIRef, Namespace
from rdflib.plugins.sparql import prepareQuery


class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self.rdf_graph = RDFGraph()
        self.ns = Namespace("http://example.org/")

    def add_node(self, node_id, properties=None):
        self.graph.add_node(node_id)
        if properties:
            for key, value in properties.items():
                self.graph.nodes[node_id][key] = value
                self.rdf_graph.add((URIRef(self.ns[node_id]), URIRef(self.ns[key]), Literal(value)))

    def add_edge(self, source, target, properties=None):
        self.graph.add_edge(source, target)
        if properties:
            for key, value in properties.items():
                self.graph[source][target][key] = value
                self.rdf_graph.add((URIRef(self.ns[source]), URIRef(self.ns[key]), URIRef(self.ns[target])))

    def get_node_properties(self, node_id):
        return self.graph.nodes[node_id]

    def get_edge_properties(self, source, target):
        return self.graph[source][target]

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

    def visualize(self):
        net = Network(notebook=True, height="500px", width="100%")
        net.from_nx(self.graph)
        return net

    def export_to_networkx(self):
        return self.graph.copy()

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
