# {KG + LLM} chatbot

There are tools/agents/plugins to connect with SQL db or tables etc. Although connections with Graph db or converting text to Graph-QLs will come soon, but the core question remains, do you have good-well-formed data or knowledge in proper structure like Knowledge Graphs (KG).

This repo explores this further. Wish to build that end-to-end App for fine-tuning LLMs on Knowledge Graphs generated on own corpus.

Graphs, actually the Knowledge Graphs, would be essential for **Sarvadnya (All-Knowing)** chatbot. So, with LangChain as LLM front-end for NLU and NLG, LLaamaIndex can be the data ingestion or back-end KG db store. Planning to leverage the combo for open-source stack: LangChain (LLM with OpenAI or HuggingFace), LLaamaIndex (Data Connectors) on Knowledge Graphs, where generation and querying, both can happen via LLM.

## Why {KG + LLM} chatbot?
- No one person can read a million articles comping every day in Pubmed. So is the velocity with legal judgments, research papers in any domain. How to comprehend, at least superficially if not fully,  semantically? At least narrow down, ie filter to get to some core documents for further manual study. Use AI ML in NLP for Information Retrieval and Text Mining techniques. Applications vertically are in many domains but horizontally for Automation for QnA and Hypothesis Generation.
- Chatbot would be the front-end for Knowledge Graph for "WoW!!" experience. Chatbot is Turing real AI even if its adding some incremental benefit.
- Solution can be domain agnostic, highly portable, for global impact and sustainable forever.
- Application: Health-care for Sr Citizen via Voice is the next best thing

## Contours
- Given text corpus (any domain: legal cases, medical papers, research articles, etc)
- Build Knowledge Graph with nodes/edges populated using (more info at [KaaS](./KaaS.md):
	- Linguistic features: POS/NER tags, Topic Models, etc
	- Domain Knowledge: Ontologies, dictionaries
- All thru configurations
- Able to fire queries to find relations, paths between concepts, sub-graphs, similarity, anomaly detection, etc.
- LangChain based chatbot as front end for NLU (Natural Language Understanding) and NLG (Natural Language Generation)

<!-- 
## Application: Elderly care
- Elderly can not type at normal chatbot, ie on mobile (WhatsApp) or Web (messenger), they are technologically challenged.
- Need voice command based system. It need not process complex linguistics of literature quality, but a simple, unambiguous, voice command system.
- Single, elderly, living alone is the target persona. Needs daily routine check, medication reminder, ordering necessary items, basic first level medical assistance, do some entertainment, help call doctor/relatives, appointments etc.
- Hindi, vernacular for wider reach, global impact, issue I care and seeing with parents.
-->

## Approaches
- Open source: [LangChain](https://python.langchain.com/en/latest/index.html) + [LLaamaIndex](https://github.com/jerryjliu/llama_index) + Knowledge Graphs
- Google Cloud: End-to-end VertexAI MLOps, easy deployment, for enterprise internal solution, with Neo4j Auro DB for Knowledge Graphs.



## Another (Stretch) Goal: Geometric Deep Learning
- Bring in Geometric Deep Learning (GDL) on the side of Knowledge Graphs. 
- GDL extends usual Deep Learning to non-manifold (variable size, networks) and tries to apply neural networks like Transformers on it.
- Imagine, a query comes not to fetch the data from knowledge graph (ie Descriptive) but to predict something based on KG (ie Predictive). 
- We will need Agents to build models and do the inference for the query.
