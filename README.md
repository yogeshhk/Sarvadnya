# Sarvadnya (सर्वज्ञ), an All-Knowing Chatbot!!

Chatbots can be real WoW!! The recent evidence is: ChatGPT. Now that they are more human-like with the latest LLMs (Large Language Models). But these LLMs are Pretrained on their own (HUGE) data. Mere mortals don't have any ways ($$, time, expertise) to train own LLMs. RAG and/or Fine-tuning is the way out for Domain Adaptation ie. LLMs answering on your corpus. This repo is a collection of various [PoCs (Proof-of-Concepts)](./src/README.md) to interface custom data using LLMs. 

A few other topics are (or can be) part of this repo is to build
- Indic-languages models, some notes [here](./references/README_AABB.md)
- 3D World Simulations, Agents, some notes [here](./references/README_3DVerse.md)
- Knowledge Graphs Generation, some notes [here](./references/README_GenKG.md) 
- Signal Processing, some notes [here](./references/README_SignalProcessing.md)
- Drones, UAV Image Processing, Shynakshi [here](./references/README_Shyenakshi.md)
- Floor Plan Segmentation [here](./references/README_FloorPlan.md)

# What?

## PoCs Projects
- Prep chatbots of various modalities, use cases and domains, diff datasets
- Prep videos, write Medium Posts (GDE/TH), LinkedIn posts, Youtube channel 

# Modes
- Retrieval Augmented Generation (RAG) on own data
- Fine-tuning LLMs with own data using LoRA etc

## RAG
- When?: {less, streaming, private} data and less {compute, money, expertise}
- What?:
	- on knowledge graphs, more grounding
	- tabular financial data, representation and similarity
	- midcurveNN Geometric serialization and retrieval
	- active loop idea of fine-tuning your data
	- Langchain and Llamaindex with any new LLM


## Fine-Tuning
- When? Sufficient curated date is available, not a whole lot though, in a batch (not running) state
- What: Instead of unstructured text (input prompts) to unstructured text (output response), more value is in prompt to structured output, such as :
	- text2json: many enterprises such as financial companies.
	- text2cypher: for graph databases, from Neo4j, like Langchain implementation by Tomaz Britanic
	- text2SQL: classical case, many pro solutions available, study them, follow them, for other QLs
	- text2Manim: Maths Animation, dataset available, see if generated video can be shown in the same streamlit page
	- text23DJS: Good for 3D+LLM+Agents like Metamorph from Nvidia, Geometry or shape representation as text, is the key
	- textGraph2textGraph: MidcurveNN if we get Graph representation as text, right.
	
- Here, key would be robust post-processing and evaluation as the response needs to be near perfect, no scope of relaxation even in syntax or format.

## Tech Stacks
- Enterprise: Google Doc AI, Vertex AI, Microsoft Azure Language AI Services
- Open Source: Langchain (Serve/Smith/Graph), HuggingFace, Streamlit for UI


## Bottom-line
- `Not looking for Success, but Wonder!!`
- तमसो मा ज्योतिर्गमय : From Dark (hidden in text data) to Light (insights)


## Folks to Follow
- Abhinav Kimothi, RAG Expert: [LinkedIn](https://www.linkedin.com/in/abhinav-kimothi/?originalSubdomain=in), [Projects Portfolio](https://www.datascienceportfol.io/abhinavkimothi), [Website](https://linktr.ee/abhinavkimothi),  [Medium](https://medium.com/@abhinavkimothi), [LinkedIn Articles](https://www.linkedin.com/in/abhinav-kimothi/recent-activity/articles/), [LinekdIn Posts](https://www.linkedin.com/in/abhinav-kimothi/recent-activity/all/), [Company](https://www.yarnit.app/)
- Pradip Nichite, Freelancing Expert: [LinkedIn](https://www.linkedin.com/in/pradipnichite/), [Projects Portfolio](https://www.aidemos.com/), [Blog](https://pradipnichite.hashnode.dev/), [Youtube](https://www.youtube.com/channel/UC3-uyUX8s536lUkrWwYvfDg), [LinekdIn Posts](https://www.linkedin.com/in/pradipnichite/recent-activity/all/), [Company](https://www.futuresmart.ai/)
- Sahar Mor: [LinkedIn](https://www.linkedin.com/in/sahar-mor/), [Blogs](https://www.aitidbits.ai/)

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
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997v1)


## Disclaimer:
Author (yogeshkulkarni@yahoo.com) gives no guarantee of the results of the program. It is just a fun script. Lot of improvements are still to be made. So, don’t depend on it at all.
