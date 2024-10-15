from pyvis import network as net

# g=net.Network(height='400px', width='50%',heading='')
# g.add_node(1)
# g.add_node(2)
# g.add_node(3)
# g.add_edge(1,2)
# g.add_edge(2,3)
# # g.show('example.html')
# g.write_html('example.html', open_browser=True, notebook=False)

import networkx as nx
from pyvis import network as net

# Create a NetworkX graph
G = nx.Graph()

# Add nodes
G.add_node("1", label="Node 1")
G.add_node("2", label="Node 2")
G.add_node("3", label="Node 3")

# Add edges
G.add_edge("1", "2", weight=2)
G.add_edge("2", "3", weight=1)

# Create a PyVis network from the NetworkX graph
g = net.Network(height='400px', width='50%', heading='', notebook=True)
g.from_nx(G)

# Show the network in the browser
g.write_html('example.html', open_browser=True, notebook=False)