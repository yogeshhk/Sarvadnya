import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any
from streamlit_agraph import Node as AgraphNode, Edge as AgraphEdge

@dataclass
class Node:
    id: str
    label: str
    size: int
    color: str
    shape: str

    def to_dict(self):
        return asdict(self)

    def to_agraph_node(self):
        return AgraphNode(**self.to_dict())

@dataclass
class Edge:
    source: str
    target: str
    type: str
    color: str
    width: float

    def to_dict(self):
        return asdict(self)

    def to_agraph_edge(self):
        return AgraphEdge(**self.to_dict())

class GraphConstructor:
    NODE_COLORS = {
        'I': "#D4A5A5",
        'II': "#9D7E79",
        'III': "#614051",
        'IV': "#A26769"
    }

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.json_data = self._load_json()
        self.nodes, self.edges, self.node_dict, self.edge_data = self._construct_graph()

    def _load_json(self) -> Dict:
        with open(self.json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _construct_graph(self) -> Tuple[List[Node], List[Edge], Dict[str, Any], List[Tuple[str, str]]]:
        nodes = []
        edges = []
        
        node_dict = {node['data']['id']: node['data'] 
                     for node in self.json_data['elements']['nodes']}
        
        for node_id, node_data in node_dict.items():
            pada = node_id.split('.')[0]
            nodes.append(Node(
                id=node_id,
                label=node_id,
                size=30,
                color=self.NODE_COLORS.get(pada, "#614051"),
                shape="dot"
            ))
        
        edge_data = [(edge['data']['source'], edge['data']['target']) 
                     for edge in self.json_data['elements']['edges']]
        
        for source, target in edge_data:
            edges.append(Edge(
                source=source,
                target=target,
                type="STRAIGHT",
                color="#614051",
                width=1
            ))
        
        return nodes, edges, node_dict, edge_data

    def get_agraph_nodes_and_edges(self):
        agraph_nodes = [node.to_agraph_node() for node in self.nodes]
        agraph_edges = [edge.to_agraph_edge() for edge in self.edges]
        return agraph_nodes, agraph_edges

    def save_changes(self, node_id: str, field: str, new_value: str) -> None:
        for node in self.json_data['elements']['nodes']:
            if node['data']['id'] == node_id:
                node['data'][field] = new_value
                self.node_dict[node_id][field] = new_value
                break
        
        self._save_json()

    def get_connected_nodes(self, selected_node: str) -> List[str]:
        return [target for source, target in self.edge_data if source == selected_node]

    def add_connection(self, source: str, target: str) -> None:
        if (source, target) not in self.edge_data:
            self.edge_data.append((source, target))
            self.edges.append(Edge(
                source=source,
                target=target,
                type="STRAIGHT",
                color="#614051",
                width=1
            ))
            self.json_data['elements']['edges'].append({
                'data': {
                    'source': source,
                    'target': target
                }
            })
            self._save_json()

    def remove_connection(self, source: str, target: str) -> None:
        if (source, target) in self.edge_data:
            self.edge_data.remove((source, target))
            self.edges = [edge for edge in self.edges if not (edge.source == source and edge.target == target)]
            self.json_data['elements']['edges'] = [
                edge for edge in self.json_data['elements']['edges']
                if not (edge['data']['source'] == source and edge['data']['target'] == target)
            ]
            self._save_json()

    def _save_json(self) -> None:
        with open(self.json_file_path, 'w', encoding='utf-8') as file:
            json.dump(self.json_data, file, ensure_ascii=False, indent=2)

    def get_all_node_ids(self) -> List[str]:
        return list(self.node_dict.keys())