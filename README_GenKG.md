# Generating Knowledge Graphs (KG)

This repository provides a comprehensive solution for generating Knowledge Graphs from various data sources. Knowledge Graphs are powerful tools for organizing and representing information in a structured and machine-readable format, enabling advanced analytics, knowledge discovery, and decision-making. Mastery of KG can be portable across industries such as Legal, Medical, Financial, etc. This skill may/will bring sustainable opportunities for Consultancy, building pay-per-use micro SaaS by solo-prenours

## Reasoning
- Ikigai: Word needs, ready to pay, I am good at, I like (this automation for sure)
- Specific knowledge is IR + KG + NLP + LLM + ML + DL unattainable
- Projects: GSOC mentor, RAG KG demos, talks {GDE, Ninja, MVP} 

## Product Specs: 
- [{KG + LLM} chatbot](https://medium.com/technology-hits/specs-for-chatbot-on-knowledge-graph-using-large-language-models-dedcff0ab553) Building LLM based Chatbot on Knowledge Graph
- [Knowledge as a Service (KaaS)](https://medium.com/technology-hits/specs-for-knowledge-as-a-service-kaas-project-9e2d9a7e0775) Building Knowledge Graph from Text and serving it as a Service
ght (insights)


## Usage

### Data Extraction
1. Implement data extraction functions to retrieve data from various sources (e.g., databases, CSV files, web APIs).
2. Preprocess the data as needed, such as cleaning, normalizing, and transforming it into a format suitable for Knowledge Graph generation.
3. Use traditional regex, or ML based CRF++, or DL based LSTM or LLM based models (Prompts, Fine-tuned) to do IR (Information Retrieval), or basically NER(Named Entity Recognition) and RE (Relationship Extraction).

### Knowledge Graph Generation
1. Define the ontology or schema for your Knowledge Graph, specifying the entities, relationships, and properties.
2. Use the `rdflib` library to create RDF triples based on the extracted data and the defined ontology.
3. Store the RDF triples in a triplestore or a graph database for efficient querying and storage.
4. Storage can be in open source DBpedia or commercial Neo4j.

### Visualization
1. Leverage libraries like `networkx` and `matplotlib` to visualize the generated Knowledge Graph.
2. Provide options for customizing the visualization, such as node and edge styles, layout algorithms, and filtering.

## Advantages RAG on KG
1. Properitary data, IP
2. Can do by direct KG similary based query or by first converting to cypher or SPARQL
3. Less Hallucination
4. More debug-ability if things go wrong, compared to plain text

## Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.


## Folks to Follow
- 


## References
- [Knowledge Graphs - Foundations and Applications (OpenHPI 2023) - ISE FIZ Karlsruhe](https://www.youtube.com/playlist?list=PLNXdQl4kBgzubTOfY5cbtxZCgg9UTe-uF)
- [Linked Data Engineering - OpenHPI 2016 -ISE FIZ Karlsruhe](https://www.youtube.com/playlist?list=PLNXdQl4kBgzuqG2lJQWER1D0VnXTkqK3v)
- [Node Classification on Knowledge Graphs using PyTorch Geometric](https://www.youtube.com/watch?v=ex2qllcVneY)
- [Geometric Deep Learning](https://www.youtube.com/playlist?list=PLn2-dEmQeTfSLXW8yXP4q_Ii58wFdxb3C)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [Machine and Language Learning Lab IISc](http://malllabiisc.github.io/)
- [Semantic Web India](http://www.semanticwebindia.com/) Enables organizations generate value from Data using AI, Knowledge Discovery
- [Cambridge Semantics](https://cambridgesemantics.com/)
- [Kenome](https://www.kenome.io/) Partha Talukdar. Helping enterprises make sense of dark data using cutting-edge Machine Learning, NLP, and Knowledge Graphs.
- [Knowledge graphs](https://www.turing.ac.uk/research/interest-groups/knowledge-graphs)
- [Geometric Deep Learning)[https://geometricdeeplearning.com/]
- [Learning on Graphs Conference](https://www.youtube.com/@learningongraphs/videos)
- [Group Equivariant Deep Learning (UvA - 2022)](https://www.youtube.com/playlist?list=PL8FnQMH2k7jzPrxqdYufoiYVHim8PyZWd)


## Disclaimer:
Author (yogeshkulkarni@yahoo.com) gives no guarantee of the results of the program. It is just a fun script. Lot of improvements are still to be made. So, don’t depend on it at all.


## License
This project is licensed under the [MIT License](LICENSE).