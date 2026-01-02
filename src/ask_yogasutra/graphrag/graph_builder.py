import json
import networkx as nx
from rdflib import Graph as RDFGraph, Literal, URIRef, Namespace
from rdflib.plugins.sparql import prepareQuery
import unittest
from typing import List, Tuple, Optional

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
    
    def get_node_by_property(self, property_name: str, property_value: str) -> List[str]:
        """Get nodes that match a specific property value."""
        matching_nodes = []
        for node, data in self.graph.nodes(data=True):
            if property_name in data and data[property_name] == property_value:
                matching_nodes.append(node)
        return matching_nodes

    def get_neighbors_with_distance(self, node_id: str, max_distance: int = 2) -> dict:
        """Get neighbors within a certain distance from the node."""
        if node_id not in self.graph:
            return {}
        return dict(nx.single_source_shortest_path_length(self.graph, node_id, cutoff=max_distance))


class TestGraphBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load actual graph data once."""
        cls.json_file = 'graph_small.json'
        with open(cls.json_file, 'r', encoding='utf-8') as f:
            cls.test_data = json.load(f)
    
    def setUp(self):
        """Set up test fixtures with real data."""
        self.builder = GraphBuilder()
        self.builder.import_data(self.test_data)
        
    def test_get_node_properties(self):
        """Test retrieving node properties from loaded data."""
        nodes = self.test_data['elements']['nodes']
        if nodes:
            node_id = nodes[0]['data']['id']
            properties = self.builder.get_node_properties(node_id)
            self.assertIn('id', properties)
            self.assertEqual(properties['id'], node_id)
    
    def test_get_connected_nodes(self):
        """Test getting connected nodes from real graph."""
        edges = self.test_data['elements']['edges']
        if edges:
            source = edges[0]['data']['source']
            connected = self.builder.get_connected_nodes(source)
            self.assertIsInstance(connected, list)
        
    def test_get_all_node_ids(self):
        """Test getting all node IDs."""
        node_ids = self.builder.get_all_node_ids()
        expected_count = len(self.test_data['elements']['nodes'])
        self.assertEqual(len(node_ids), expected_count)
        
    def test_get_all_tags(self):
        """Test getting all unique tags from real data."""
        tags = self.builder.get_all_tags()
        self.assertIsInstance(tags, list)
        
    def test_sparql_query(self):
        """Test SPARQL query functionality."""
        query = """
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o .
        }
        LIMIT 5
        """
        results = self.builder.sparql_query(query)
        self.assertIsNotNone(results)
        
def run_tests():
    """Run all graph builder tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphBuilder)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

# Update __main__ section
if __name__ == "__main__":
    print("Running GraphBuilder tests...")
    success = run_tests()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")