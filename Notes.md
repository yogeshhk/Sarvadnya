# Notes
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
