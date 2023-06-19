# Knowledge as a Service (KaaS)

## Why

'Data is the new oil' is ok, but raw oil is useless unless it is refined for end purpose. So, refining or curating data is the key, mainly for downstream applications such as training Machine/Deep Learning algorithms.

Algorithms is commodity, (curated) data is gold, a refined oil. Plus it is hard to keep up with advancements in algorithms, not manageable, but need for curated data is high and going to be there forever, an IKIGAI.

Data curation companies are in abundance and each ML company anyways does it internally, but can it be made as a external, 
paid Service? On which datasets, in which usable-intermediate form? where to get raw datasets? privacy or ownership issues?


Data curation can be vast, labor intensive,  employment generation for review, multidisciplinary, business or social, etc. Need to build niche.

Build KG from texts. Its a Specific knowledge : Specialize is graphs, knowledge graphs, expertise in nlp for extraction from text,

KG itself is useful for insights using GQLs, but can also be used for predictions. But this is domain specific and labor/review intensive, especially the relationship extraction part.
Another thought is not to serve curated data as is but in binary/vector form, so hard to reverse engineering and privacy preserving.
So basically vector db access is what you are selling,  

IKIGAI:
- What world needs: ready-to-use data,for ML/DL
- What world is ready to pay for? easy access to curataed data, like HuggingFace or Kaggle
- What you like? Building KGs
- What you are good at? working in NLP/LLMs

## Notes

-Formalizing Hypothesis Virtues in Knowledge Graphs
	- Find subgraphs which have more potential to find hypothesis, thus avoiding lots of noise and redundancy
	- These potential Discovery Informatics regions are found using Virtues.		
	-  A claim of a hypothesis H is a simple (i.e., acyclic) path in the
graph H.
	- Virtues of hypothesis: conservatism, modesty, simplicity, generality and refutability.
			- Conservatism: minimize risk of error by going too far from current state of the art solution in one step. Aggregation valies over the path should be max.
			- Modesty: minimizes risk of wrong/redundant claims. Ratio of total claims with Hypothesis claims should be large.
			- Simplicity: simplify world view even if underlying claim is complex. Ratio of complexity measures of Universe given Hypotheses with Hypothesis.
			- Generality: more phenomenon it can predict and explain. A number of explanations (i.e., claims) the hypothesis H can provide for ‘out-of-scope’ phenomena (i.e., vertices) in the
U \ H graph.
			- Refutability: should be falsifiable. How much reduction is possible in number of claims to core ones.
		- Individual measures can be combined to get an overall score.
	- Original graph, which is purely NLP-extraction based, can be refined so that it carries more potentially useful hypotheses.
	- Refinement is done evolutionarily ie by genetic algorithm
	
	[YK]: The approach in the paper appears domain agnostic, with specific measures to highlight interesting regions in knowledge graph
	
- Deep Learning-Based Knowledge Graph Generation for COVID-19
	- Deep Learning, unsupervised, co-BERT based Relation Extraction.
	- Using BERT, a mask covers a predefined word and an entity, and the relationship between the words is determined through the self-attention weight. 
	- Approaches for constructing knowledge graphs using BERT
		- Construct a knowledge graph by masking specific entities, predicting them correctly, and extracting them as knowledge.
		- Iput existing knowledge bases into BERT and generates a corresponding knowledge graph that can be expanded continuously
	- To extract the relation between the head entity and tail entity using the attention weight is used. The word most related to the two entities is searched for in all attention maps, and the number
of occurrences is counted for each word
	
	
- Query Driven Hypothesis Generation for Answering Queries over NLP Graphs
	- Store NLP as graph with entities at nodes and relations at edges
	- Conjugate	queries recall is roughly product of individual relations recalls
	- Each hypothesis set from the hypothesis generator contains hypotheses in the form of RDF statements, which, if added to the primary extraction NLP graph, would provide a new answer to the original query. 
	-  Each hypothesis checker reports its confidence that the hypothesis holds and, when possible, gives a pointer to a span of text in the target corpus that supports the hypothesis (the provenance).
	
- Building Knowledgeable Machines - Partha Talukdar, Assistant Professor IISc https://www.youtube.com/watch?v=z9vzmXTLpKs
	- Background knowledge is key to intelligent decision making

- Training Series: Create a Knowledge Graph: A Simple ML Approach https://www.youtube.com/watch?v=LSCzMOcqgq8
	
- An Introduction to Knowledge Graphs http://ai.stanford.edu/blog/introduction-to-knowledge-graphs/

	- A knowledge graph is a directed labeled graph in which we have associated domain specific meanings with nodes and edges.

	- Definition:  A directed labeled graph is a 4-tuple $G = (N, E, L, f)$, where N is a set of nodes, $E \subseteq N \times N$ is a set of edges, $L$ is a set of labels, and $f: E \rightarrow  L$, is an assignment function from edges to labels. An assignment of a label $B$ to an edge $E=(A,C)$ can be viewed as a triple $(A, B, C)$

	- A well-documented list of relations in Schema.Org, also known as the relation vocabulary

	- A recent version of Wikidata had over 90 million objects, with over one billion relationships among those objects. Wikidata makes connections across over 4872 different catalogs in 414 different languages published by independent data providers. As per a recent estimate, 31% of the websites, and over 12 million data providers are currently using the vocabulary of Schema.Org to publish annotations to their web pages.

	- An ontology is a formal specification of the relationships that are used in a knowledge graph.
	
	- To make the internet more intelligent, the World Wide Web Consortium (W3C) standardized a family of knowledge representation languages that are now widely used for capturing knowledge on the internet. These languages include the Resource Description Framework (RDF), the Web Ontology Language(OWL), and the Semantic Web Rule Language (SWRL).
 
- Hypothesis Generation with AGATHA : Accelerate Scientific Discovery with Deep Learning | AISC https://www.youtube.com/watch?v=Q2po1QZONIg
	- Moliere 2017
		- Process:
			- Construct Semantic Network
			- Shortest Path Queries
			- Topic Modeling
		- Limitations:
			- Minutes per query
	- Agatha
		- Process:
			- Parse MEDLINE
			- Construct Semantic Network
			- Train Embedding and model
			- Validate through ranking
			- Removed Heuristics
			- Added data driven insights
			- Speed-up queries
		
- CS520: Knowledge Graphs Seminar (Spring 2020) https://www.youtube.com/playlist?list=PLDhh0lALedc7LC_5wpi5gDnPRnu1GSyRG

- CS 520 2021 Knowledge Graphs https://web.stanford.edu/class/cs520/