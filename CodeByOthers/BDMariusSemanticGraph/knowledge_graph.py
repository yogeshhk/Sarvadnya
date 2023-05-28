from relation import Relation
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph:

    __relations: [Relation]
    __graph: nx.Graph
    __colors: {}

    def __init__(self, relations):
        self.__relations = relations
        self.__graph = nx.Graph()
        self.__colors = {}

    def build(self):
        for relation in self.__relations:
            self.__graph.add_node(relation.getHypernym())
            self.__colors[relation.getHypernym()] = '#e34234'
            self.__graph.add_node(relation.getHyponym())
            self.__colors[relation.getHyponym()] = '#009966'
            self.__graph.add_edge(relation.getHypernym(), relation.getHyponym())

    def show(self):
        pos = nx.spring_layout(self.__graph)
        plt.figure()
        colorMap = []
        for node in self.__graph.nodes:
            colorMap.append(self.__colors[node])
        nx.draw(self.__graph, pos, edge_color='black', width=1, linewidths=1,
                node_size=500, node_color=colorMap, alpha=0.9,
                labels={node: node for node in self.__graph.nodes()})
        plt.axis('off')
        plt.show()