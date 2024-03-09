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

## MicroSaaS

- Learn some end-to-end hosting platform (LangServe? Langchian with Azure AI?)
- Convert above demos to have user input (disclaimers, limited uploads, $$)

## Why me?
- Need to be 'creating', time-based-jobs can't scale, its renting, need to leverage intellect
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

## Manifestation
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

## Why LangChain? Unofficial Developer Advocate
- Local (secure), no over-the-net API/web calls
- Open source, Free via HuggingFace, Contrib possible
- PoC to Prod, end-to-end
- Python!! end-to-end, with Streamlit as UI
- Huge support, community, opportunities
- Coach: write/talk about it via Medium Stories, Webinars, LinkedIn posts (Mvp ++, advocu ++)
- Passive MicroSaaS income, pay per use, Integrations

## LangChain MicroSaaS
1. **Focused Development**: Micro SaaS businesses focus on serving a niche market or a specific customer segment with a highly targeted software solution[^10^]. This allows for focused development and targeted marketing⁹.

2. **Cost-Effective**: Micro SaaS businesses operate with minimal resources, leveraging cloud infrastructure and automation tools to streamline operations and keep costs low[^10^]. RAG offers an affordable, secure, and explainable alternative to general-purpose LLMs, drastically reducing the likelihood of hallucination⁴.

3. **Customized Solutions**: RAG allows businesses to achieve customized solutions while maintaining data relevance and optimizing costs⁶. By adopting RAG, companies can use the reasoning capabilities of LLMs, utilizing their existing models to process and generate responses based on new data⁴.

4. **Integration with LangChain**: LangChain is a framework designed to simplify the creation of applications using LLMs¹. It can dynamically connect different systems, chains, and modules to use data and functions from many sources, like different LLMs¹. This allows businesses to develop language model-powered software applications that can carry out various activities, including code analysis, document analysis, and summarization².

5. **Data-Aware and Agentic**: LangChain is data-aware and agentic, enabling connections with various data sources for richer, personalized experiences³. This allows for better interoperability across the board, offering various valuable tools that allow businesses to connect to different vendors (including other LLMs) and integrations with a comprehensive collection of open-source components¹.

6. **Access to Various LLM Providers**: LangChain offers access to LLMs from various providers like OpenAI, Hugging Face, Cohere, AI24labs, among others¹. These models can be accessed through API calls using platform-specific API tokens, allowing developers to leverage their advanced capabilities to build as they see fit¹.

7. **Recurring Profits and Low Risk**: With their recurring profits, fewer capital needs, low risk, dedicated customers and minimal operating expenses, Micro SaaS has started attracting many entrepreneurs towards them in recent years¹⁴.

8. **Stable Recurring Income**: Micro-SaaS businesses are usually location-independent and can be a source of stable recurring income once the product has achieved a loyal user base¹¹.


Sources: 
(1) What is Micro SaaS And How to Create One In 2024. https://bufferapps.com/blog/what-is-micro-saas/.
(2) Complete Guide to Micro-Saas: Build a Profitable Business.. https://blog.payproglobal.com/micro-saas-guide.
(3) RAG and LLM business process automation: A technical strategy. https://blog.griddynamics.com/retrieval-augmented-generation-llm/.
(4) Retrieval Augmented Generation using Azure Machine Learning prompt flow .... https://learn.microsoft.com/en-us/azure/machine-learning/concept-retrieval-augmented-generation?view=azureml-api-2.
(5) What is LangChain: How It Enables Businesses to Do More with LLMs. https://www.bluelabellabs.com/blog/what-is-langchain/.
(6) LangChain: A New Era of Business Innovation - Medium. https://medium.com/@tvs_next/langchain-a-new-era-of-business-innovation-7207a44382c9.
(7) What is LangChain? A Beginners Guide With Examples - Enterprise DNA Blog. https://blog.enterprisedna.co/what-is-langchain-a-beginners-guide-with-examples/.
(8) Top 25 Profitable Micro SaaS Business Ideas in 2022 - StartupTalky. https://startuptalky.com/micro-saas-business-ideas/.
(9) Building a Micro-SaaS: Best Tools and Platforms In 2022 - Saastitute. https://www.saastitute.com/blog/building-a-micro-saas-best-tools-and-platforms.
(10) Improve LLM responses in RAG use cases by interacting with the user. https://aws.amazon.com/blogs/machine-learning/improve-llm-responses-in-rag-use-cases-by-interacting-with-the-user/.
(11) An introduction to RAG and simple/ complex RAG - Medium. https://medium.com/enterprise-rag/an-introduction-to-rag-and-simple-complex-rag-9c3aa9bd017b.
(12) Concept of RAG (Retreival-Augmented Generation) in LLM. https://blog.devgenius.io/concept-of-rag-retreival-augmented-generation-in-llm-4f878251b4d1.
(13) How To Build a Profitable Micro-SaaS Business in 2024 - BufferApps. https://bufferapps.com/blog/how-to-build-a-micro-saas/.
(14) Top 10 Micro SaaS Ideas To Build a Profitable Business in 2024. https://controlhippo.com/blog/micro-saas/.

