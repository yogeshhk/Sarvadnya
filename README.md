# Sarvadnya (सर्वज्ञ), an All-Knowing Micro SaaS!!

Embarking on the journey to master the powerful and sought-after paradigm of RAG, along with multi-modal fine-tuning techniques and Knowledge Graphs, is a rewarding endeavor. 

Consider maintaining MidcurveLLM as a parallel R&D. 

The combination of LLM and KG is an IKIGAI - a concept that the world needs, is willing to pay for, and something you are good at and enjoy. This aligns with Naval’s suggestion of cultivating ‘Specific Knowledge’ - a unique skill set that is untrainable and possessed by few.

Chatbots can be real WoW!! The recent evidence is: ChatGPT. Now that they are more human-like with the latest LLMs (Large Language Models). But these LLMs are Pretrained on their own (HUGE) data. Mere mortals don't have any ways ($$, time, expertise) to train own LLMs.
Some do have facility to get fine-tuned on custom corpus, but limited. Custom fine tuning of text documents is being provided by many. 

This repo is a collection of various [PoCs (Proof-of-Concepts)](./src/README.md) to interface custom data using LLMs.

Stretch (RnD) goals: 
- [{KG + LLM} chatbot](https://medium.com/technology-hits/specs-for-chatbot-on-knowledge-graph-using-large-language-models-dedcff0ab553) Building LLM based Chatbot on Knowledge Graph
- [Knowledge as a Service (KaaS)](https://medium.com/technology-hits/specs-for-knowledge-as-a-service-kaas-project-9e2d9a7e0775) Building Knowledge Graph from Text and serving it as a Service
- LLM models for Indic (especially Sanskrit) languages. Here is collection of similar efforts going on [Awesome AI By Bharat (AABB)](./README_AABB.md)

## Main Modes for Custom LLM/Chatbot
- Fine-tuning LLMs with own data using LoRA etc
- Retrieval Augmented Generation (RAG) on own data

## RAG

- WHY?: World needs, ready to pay, I am good at, and I like it making wow chatbots
- Domain:
	- on knowledge graphs, more grounding
	- tabular financial data, representation and similarity
	- midcurveNN Geometric serialisation and retrieval
	- active loop idea of fine-tuning your data
	- langchain and llamaindex with any new llm
	- bharat gpt, bhashini with sanskrit, do prototype on arthashastra principles
- Specific Knowledge - LLMs, Graphs, Sanskrit 

## Chatbot Pathways
- Enterprise: Google Cloud: Gen AI, Doc AI, Vertex AI: Skills Boost paths, Professional ML Certification
- Open Source: Langchain, HuggingFace, Streamlit: Custom fine-tuned models

## Why LangChain based Implementations ?:
- Local (secure), no over-the-net API/web calls
- Open source, Free via HuggingFace
- Python!! end-to-end, with Streamlit as UI
- Huge support, community, opportunities


## Publications so far
- [SaaS LLM](https://medium.com/google-developer-experts/saasgpt-84ba80265d0f)
- [AskAlmanackBot](https://www.linkedin.com/feed/update/urn:li:ugcPost:7049347127029698560/)
- [GST FAQs Bot](https://medium.com/google-cloud/building-a-gst-faqs-app-d8d903eb9c6)

## References
- [LangChain How to and guides](https://www.youtube.com/playlist?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)
- [Building the Future with LLMs, LangChain, & Pinecone](https://www.youtube.com/watch?v=nMniwlGyX-c)
- [LangChain for Gen AI and LLMs - James Briggs](https://www.youtube.com/playlist?list=PLIUOU7oqGTLieV9uTIFMm6_4PXg-hlN6F)
- [Finetuning GPT-3 David Shapiro ~ AI](https://www.youtube.com/playlist?list=PLV3Fr1UUO9bFg3tKw_-6djIhgId1z74JU)
- [Build overpowered AI apps with the OP stack (OpenAI + Pinecone)](https://www.youtube.com/watch?v=-dZrNj2mVHo)
- [Learn about AI Language Models and Reinforcement Learning Kamalraj M M](https://www.youtube.com/playlist?list=PLbzjzOKeYPCpp3NCeQioevM0YpZa5VqcS)
- [GPT-4 & LangChain Tutorial: How to Chat With A 56-Page PDF Document (w/Pinecone)](https://www.youtube.com/watch?v=ih9PBGVVOO4)
- [LangChain - Data Independent](https://www.youtube.com/playlist?list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5)
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
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997v1)
Key ideas the paper discusses to make your RAG more efficient [Ahmed BESBES](https://www.linkedin.com/posts/ahmed-besbes-_machinelearning-llms-datascience-activity-7147161560791019520-uz97):
🗃️ Enhance the quality of indexed data by removing duplicate/redundant information and adding mechanisms to refresh outdated documents
🛠️ Optimize index structure by determining the right chunk size through quantitative evaluation
🏷️ Add metadata (e.g. date, chapters, or subsection) to the indexed documents to incorporate filtering functionalities that enhance efficiency and relevance
↔️ Align the input query with the documents by indexing the chunks of data by the questions they answer
🔍 Mixed retrieval: combine different search techniques like keyword-based and semantic search
🔄 ReRank: sort the retrieved documents to maximize diversity and optimize the similarity with a « template answer »
🗜️ Prompt compression: remove irrelevant context
💡 HyDE: generate a hypothetical answer to the input question and use it (with the query) to improve the search
✒️ Query rewrite and expansion to reformulate the user’s intent and remove ambiguity

- [Notes from Navdeep Singh](https://www.linkedin.com/in/navdeepsingh1604/)
In my pursuit of building a Self-Service RAG platform, I grappled with the challenges of accuracy and scalability. While creating a RAG prototype is straightforward, transforming it into a production-ready, scalable, and high-performance application demands a strategic focus on key limitations.

🔥💡Below diagram distills the essential optimizations for RAG applications, offering achieving production-grade excellence in terms of performance, reliability, accuracy, and scalability.



🦄 Data Ingestion Pipeline:
✅ Collecting Data:
Harness the RAG system's power to collect and process diverse data from structured databases, trusted websites, and policy documents.
Implement unique connectors tailored to each data source, ranging from API calls to document parsing and web scraping.
✅ Data Cleaning and Metadata Enrichment:
Ensure high-quality data by eliminating inconsistencies, errors, and encoding issues.Validate data integrity and enrich metadata to enhance post-search results, offering additional contextual filters.
✅ Smart Chunking and Labeling:
Break down large documents into coherent units for improved retrieval performance.
Label data with contextual information, enhancing reliability and trust in the outcomes.
✅ Embedding Models:
Fine-tune embedding models for optimal retrieval outcomes.
Leverage specialized, use-case specific models to enhance overall performance.
✅Vector Database:
Implement a vector database for efficient retrieval and similarity search.
Experiment with indexing algorithms for granular context separation and employ Approximate Nearest Neighbor (ANN) search methodologies for scalability.
✂ Retrieval and Generation Phase:
✅Query Enhancement:
Use the LLM to iteratively optimize search queries for better results.
Employ query partitioning for complex queries.
✅ Re-ranking Technique:
Mitigate non-relevant information by employing re-ranking models based on relevance score calculation.
✅ Prompt Engineering:
Craft clear and concise prompts for accurate outputs.
Inject specificity and control into prompts to guide the LLM's generation.
Provide additional context or knowledge through prompts for improved accuracy.
I want to thank all my fellow Contributors from whom I learnt this year.

<!-- 
## Why me?
- Need to be 'creating', time-based-jobs cant scale, its renting, need to leverage intellect
- MicroSaaS: one-person company, pay-per-use service for passive income forever
- Build out-of-the-world idea, a wow automation, using NLP LLM Gen AI.
- Find pressing pain points to address, in daily use, for wider audience
- Devote time playfully to solve the problem, with peace and joy, no anxiety no fomo, no hopes of revenue but just a journey to develop something very cool
- Reasonable popularity due to Sketchnote and talks on ChatGPT or LLMs (Large Language Models), will help the spread
- Specific Knowledge: Theoretical background of NLP/LLMs due to trainings, plus, professional experience on customizing LLMs on custom data, plus common-sense software solution-ing experience for 2 decades, including engineering industries. Rare-Global-Untrainable-Leverage-Brand.
- IKIGAI: I love, I like, World needs, Can get paid
	- World needs: huge corpus, global, domain
	- Good at: ML, NLP professional experience
	- Love doing: automation, part II
	- Paid for: consult, train, passive service

^Specific Knowledge: rare, un-trainable, only through apprenticeship 

## Theme
- Theme: Automation MicroSaas
- Product: Micro SaaS, auto upgrade, Serverless, scale as you go
- Payment: Pay per use, Passive Income, forever
- Income: Passive, remote fully, global reach
- Working: Solo, remote, no team, no HR issues, salaries
- Input: scraping , docAI(GDE)->KG (neo4j)
- Output: Wow chatbot, APIs, Network effects, more connection, more $$
- Moat/Entry Barrier, IKIGAI, Sp Knowledge
- Give back: Talks, sketchnotes, Tech explanations
- Side outcomes: consultancy, open source contribution 


## Mode: MicroSaas
- Own (no team), 
- Pay per use, 
- Passive sustainable income, 
- Why: IKIGAI, Specific Knowledge, 

## Checklist: MicroSaas
- Do you have unfair advantage: 
	- Network of founders, influences, for further reach 
	- Audience: folks who want this app and can pay
	- Being early
- Start With a Problem or many problems (don’t tell me your ideas)
- Move from Problems to Solutions, easy, debuggable
- Evaluate Your Solutions
- How is Your Solution Different?
- Talk to Potential Customers
- Start Marketing Before Coding
- Build MVP
- Solves any specific need (pain point) and not anything-and-everything, 
- Is it for specific people, 1000 true (paying) fans, say $30 or $3 a month
- Is it a daily need?
 -->

## Bottom-line
- `Not looking for Success, but Wonder!!`
- तमसो मा ज्योतिर्गमय : From Dark (hidden in text data) to Light (insights)
