import json
import networkx as nx
from rdflib import Graph as RDFGraph, Literal, URIRef, Namespace
from rdflib.plugins.sparql import prepareQuery

class GraphBuilder:
    def __init__(self, json_file='data/graph.json'):
        self.graph = nx.Graph()
        self.rdf_graph = RDFGraph()
        self.ns = Namespace("http://example.org/")
        self.rdf_graph.bind("ex", self.ns)
        self.json_file = json_file
        self.positions = {}
        self.load_data()

    def load_data(self):
        try:
            with open(self.json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.import_data(data)
        except FileNotFoundError:
            print(f"File not found: {self.json_file}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {self.json_file}")

    def check_edge_exists(self, source_node_id, target_node_id):
        return self.graph.has_edge(source_node_id, target_node_id)

    def add_node(self, node_id, properties=None):
        self.graph.add_node(node_id, label=node_id)
        if properties:
            for key, value in properties.items():
                self.graph.nodes[node_id][key] = value
                self.rdf_graph.add((URIRef(self.ns[node_id]), URIRef(self.ns[key]), Literal(value)))

    def add_edge(self, source, target, properties=None):
        if source is None or target is None:
            print("add edge: one of them is None")
            return

        if self.check_edge_exists(source, target):
            print(f"add edge: already exists {source} and {target}")
            return

        self.graph.add_edge(source, target, weight=1)
        self.rdf_graph.add((self.ns[source], self.ns['connected_to'], self.ns[target]))
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
        self.graph = nx.Graph()
        self.rdf_graph = RDFGraph()
        self.ns = Namespace("http://example.org/")
        self.rdf_graph.bind("ex", self.ns)

        for node in data['elements']['nodes']:
            node_id = node['data']['id']
            self.add_node(node_id, node['data'])

        for edge in data['elements']['edges']:
            source = edge['data']['source']
            target = edge['data']['target']
            self.add_edge(source, target, edge['data'])

        if 'positions' in data:
            for node_id, pos in data['positions'].items():
                if node_id in self.graph.nodes:
                    self.positions[node_id] = (pos['x'], pos['y'])
                    self.update_node_properties(node_id, {'x': pos['x'], 'y': pos['y']})

        return len(self.graph.nodes), len(self.graph.edges)

    def get_connected_nodes(self, selected_node):
        return list(self.graph.neighbors(selected_node))

    def add_connection(self, source, target):
        self.add_edge(source, target)

    def remove_connection(self, source, target):
        self.graph.remove_edge(source, target)
        self.rdf_graph.remove((self.ns[source], self.ns['connected_to'], self.ns[target]))

    def get_all_node_ids(self):
        return sorted(list(self.graph.nodes()))

    def get_all_node_fields(self):
        fields = set()
        for node, data in self.graph.nodes(data=True):
            fields.update(data.keys())
        return sorted(list(fields))

    def save_changes(self, node_id, field, new_value):
        self.update_node_properties(node_id, {field: new_value})

    def export_to_json(self):
        data = {
            "elements": {
                "nodes": [],
                "edges": []
            },
            "positions": {}
        }

        for node, attrs in self.graph.nodes(data=True):
            node_data = {"id": node}
            node_data.update(attrs)
            data["elements"]["nodes"].append({"data": node_data})
            if 'x' in attrs and 'y' in attrs:
                data["positions"][node] = {"x": attrs['x'], "y": attrs['y']}

        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {"source": source, "target": target}
            edge_data.update(attrs)
            data["elements"]["edges"].append({"data": edge_data})

        return json.dumps(data, indent=2)

    def save_to_file(self, file_path=None):
        if file_path is None:
            file_path = self.json_file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(self.export_to_json()), f, ensure_ascii=False, indent=2)

    def get_all_tags(self):
        tags = set()
        for node, data in self.graph.nodes(data=True):
            if 'tags' in data:
                tags.update(data['tags'].split(','))
        return sorted(list(tags))

    def get_node_tags(self, node_id):
        node_data = self.graph.nodes[node_id]
        if 'tags' in node_data:
            return node_data['tags'].split(',')
        return []

    def get_node_positions(self):
        return self.positions