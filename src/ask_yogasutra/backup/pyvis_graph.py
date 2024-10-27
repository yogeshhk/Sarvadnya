# import networkx as nx
# from pyvis import network as net
#
# # Using the provided data structure
# data = {
#   "elements": {
#     "nodes": [
#       {
#         "data": {
#           "id": "I.1",
#           "Devanagari_Text": "अथ योगानुशासनम् ॥ १.१ ",
#           "Sanskrit_Text": "atha yogānuśāsanam",
#           "Word_for_Word_Analysis": "atha—now. yoga—of Yoga, or concentration, contemplation (samādhi) anuśāsanaṃ—a revised text, or explanation.",
#           "Translation_Bryant": "Now, the teachings of yoga [are presented].",
#           "Translation_Leggett": "Now the exposition of yoga",
#           "references": "Mahabhasya, I.1: atha śabdānuśāsanam https://archive.org/details/mahabhasya_of_patanjali_surendranath_dasgupta/page/n11/mode/2up?view=theater",
#           "tags": "Grammarians",
#           "Vyasa_sanskrit": "\n\nathetyayamadhikārārthaḥ. samādhiḥ. ",
#           "Vyasa_commentary": "“Now.”—This word That which pervades all these planes is called Sārvabhauma, all-pervading.",
#           "Hariharananda_Translation": "Now Then Yoga Is Being Explained.",
#           "Vivekananda_translation": "Now concentration is explained"
#         }
#       },
#       {
#         "data": {
#           "id": "I.2",
#           "Devanagari_Text": "योगश् चित्तवृत्तिनिरोधः ॥ १.२ ",
#           "Sanskrit_Text": "yogaś citta-vṛtti-nirodha",
#           "Word_for_Word_Analysis": "yogaḥ—yoga. citta—of the mind, mental. vṛtti—of the modifications, changes, various forms. nirodhaḥ—restraint.",
#           "Translation_Bryant": "Yoga is the stilling of the changing states of the mind.",
#           "Translation_Leggett": "Yoga is inhibition of the mental processes",
#           "tags": "Samkhya, Buddhism, Vṛtti",
#           "Vyasa_sanskrit": "\n\nsarvaśabdāgrahaṇātsaṃprajñāto'pi yoga ityākhyāyate. cittaṃ hi prakhyāpravṛttisthitiśīlatvāttriguṇam.  modifications.",
#           "Vacaspati_commentary": "The second a mind.”",
#           "Hariharananda_Translation": " Yoga Is The Suppression Of The Modifications Of The Mind\n[IT]:  Yoga is the inhibition of the modifications of the mind.",
#           "Vivekananda_translation": " Yoga is restraining the mind – stuff (Chitta) from taking various forms (Vrittis)."
#         }
#       }
#     ],
#     "edges": [
#       {
#         "data": {
#           "source": "I.1",
#           "target": "I.2",
#           "id": "169f6755-5a6f-4c2e-ac5e-124649bcb8d7"
#         }
#       }
#     ]
#   },
#   "positions": {
#     "I.1": {
#       "x": -1069.1906184015531,
#       "y": -300.58282774606494
#     },
#     "I.2": {
#       "x": 4592.14649995524,
#       "y": -300.58282774606494
#     }
#   }
# }
#
# # Create a NetworkX graph
# G = nx.Graph()
#
# # Create a PyVis network
# g = net.Network(height='400px', width='50%', heading='', notebook=True)
#
# # Extract nodes and edges from the data structure
# nodes = data["elements"]["nodes"]
# edges = data["elements"]["edges"]
#
# # Add nodes to the PyVis network
# for node in nodes:
#     node_id = node["data"].get("ID", node["data"].get("id"))  # Check for both "ID" and "id" keys
#     if node_id:
#         g.add_node(node_id, label=node_id)  # Use Devanagari Text for label
#     else:
#         print(f"Warning: Skipping node with missing ID in data: {node}")
#
# # Add edges to the PyVis network
# for edge in edges:
#     g.add_edge(edge["data"]["source"], edge["data"]["target"])

import networkx as nx
from pyvis import network as net

# Using the provided data structure
data = {
  "elements": {
    "nodes": [
      {
        "data": {
          "ID": "I.1",  # Assuming the actual ID key is "ID" (modify if different)
          "Devanagari_Text": "अथ योगानुशासनम् ॥ १.१ ",
          "Sanskrit_Text": "atha yogānuśāsanam",
          # ... other data ...
        }
      },
      {
        "data": {
          "ID": "I.2",  # Assuming the actual ID key is "ID" (modify if different)
          "Devanagari_Text": "योगश् चित्तवृत्तिनिरोधः ॥ १.२ ",
          "Sanskrit_Text": "yogaś citta-vṛtti-nirodha",
          # ... other data ...
        }
      }
    ],
    "edges": [
      {
        "data": {
          "source": "I.1",
          "target": "I.2",
          "id": "169f6755-5a6f-4c2e-ac5e-124649bcb8d7"
        }
      }
    ]
  },
  "positions": {
    "I.1": {
      "x": -1069.1906184015531,
      "y": -300.58282774606494
    },
    "I.2": {
      "x": 4592.14649995524,
      "y": 5921.021468776722
    }
  }
}

# Create a NetworkX graph
G = nx.Graph()

# Extract nodes and edges from the data structure
nodes = data["elements"]["nodes"]
edges = data["elements"]["edges"]

# Add nodes to the NetworkX graph, handling potential key variations
for node in nodes:
    node_id = node["data"].get("ID", node["data"].get("id"))  # Check for both "ID" and "id" keys
    if node_id:
        G.add_node(node_id, label=node_id)  # Use Devanagari Text for label
    else:
        print(f"Warning: Skipping node with missing ID in data: {node}")

# Extract source and target from each edge's data dictionary
for edge in edges:
    source = edge["data"]["source"]
    target = edge["data"]["target"]
    G.add_edge(source, target)

# Create a PyVis network from the NetworkX graph
g = net.Network(height='400px', width='50%', heading='', notebook=False)
g.from_nx(G)


# Show the network in the browser
# Show the network in the browser
g.write_html('example.html', open_browser=True, notebook=False)