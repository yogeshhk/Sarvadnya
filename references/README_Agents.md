#   कारक (Kāraka)

कारक (Kāraka) from Sanskrit means agent, doer. 'Kaaraka' is an open-source project that provides a framework for building and deploying intelligent agents powered by large language models (LLMs). These agents can be used for various tasks such as question-answering, task automation, personalized assistance, and more.

"Vertical AI Agents could be 10X bigger than SaaS" - Y Combinator

Levels of LLMs: QnA (atomic Prompts) -> Chatbot (with context) -> RAG (own data) -> Agents (Actions/Tools) -> OS (multiple functions, customizable tools)

Need to build something "WoW!!". So, not looking for Success but for Wonder!!. Think of a [Startup](./Startup.md).

## Introduction

Large Language Models (LLMs) are transformative AI models capable of generating human-like text, understanding natural language, and performing a wide range of tasks. However, deploying and interacting with LLMs can be challenging, especially when it comes to integrating them with other systems and managing their behavior.

LLM-Agents aims to simplify the process of building and deploying LLM-based agents by providing a modular and extensible framework. It abstracts away the complexities of working with LLMs and offers a set of tools and utilities for creating, configuring, and managing intelligent agents.

## Frameworks
- **LangGraph**: Open-source, inline with LangChain, good definition of workflows, conditional-edges, open-source LLMs native
- **AutoGen**: open-source (yes, though from Microsoft), assuming enterprise grade, deployable on Azure, less pattern/graph definition support, workaround to use open-source LLMs
- **CrewAI**: open-source


## Lang* Ecosystem
- **LangChain**: LangChain is an open-source framework designed to simplify building applications powered by language models. It helps developers chain prompts, interact with external data, and create context-aware applications. LangChain abstracts the complexity of managing API calls, memory, and agent decisions, reducing boilerplate code and improving maintainability.

- **LangGraph**: LangGraph is built on top of LangChain to manage agents and workflows, making it ideal for applications that involve multiple agents interacting to solve complex tasks. Use Cases: Task automation, Research assistants, Scenarios where agents collaborate and interact in cyclical workflows.

- **LangFlow**: LangFlow is a drag-and-drop interface built on top of LangChain, aimed at prototyping AI-powered applications quickly without needing to write code.

- **LangSmith**: LangSmith assists with deploying, testing, and monitoring LLM-based applications, ensuring that agents and LLM calls perform as expected.

- **LangServe**: LangServe is a tool designed for serving and deploying language model applications at scale. It facilitates the deployment of LLM-based applications in production environments with enhanced performance, scalability, and monitoring capabilities.

### Selection

1. **For Software Development Tasks:** : **Autogen**: Best framework overall for this use case due to its robust features and ease of use.
2. **For Newbies:**: **OpenAI Swarm** and **Crew AI**: Easy to get started with.  **Crew AI** is preferable due to better community support and production readiness.
3. **For Complex Tasks:** :**LangGraph**: Ideal for managing complicated pipelines and workflows.
4. **For Open-Source LLM Integration:**: **LangGraph** and **Crew AI**: Excellent support for leveraging open-source models.
5. **For Community Support:**  **Autogen**: Mature and active community, including resources like a subreddit.
6. **For Quick Deployment:**: **Crew AI**: Ready-to-go with minimal effort—just define some prompts, and it’s ready.
7. **Cost-Effective Solutions:** : **Magentic One**: Pre-packaged setup with four default agents, reducing initial cost.  
   - **OpenAI Swarm**: Another cost-effective option.

**Decision**: LangGraph (with RAG tools from LangChain, Open Source LLMs are native, deploy with LangServe, debug with LangSmith), all for Free

## Key Features to look for
- **Agent Creation**: Define agents with specific capabilities, knowledge bases, and personas using a declarative configuration approach, similar to Predibase-Ludwig. Using this config, one can instantiate any agentic framework such as Microsoft's AutoGen or Crew.AI or LangGraph.
- **Modular Design**: Easily swap out or customize different components of an agent, such as the language model, knowledge base, or decision-making strategy.
- **Multi-modal Support**: Agents can process and generate various data types, including text, images, audio, and more.
- **Monitoring and Logging**: Comprehensive logging and monitoring capabilities for tracking agent performance, debugging, and auditing.
- **Deployment Options**: Deploy agents locally, on cloud platforms, or as microservices, with support for scaling and load balancing.
- **Security and Privacy**: Built-in safeguards and controls for ensuring the responsible and ethical use of LLM-based agents.

## Applications
- **Quality Assurance** build test-case generator based on Requirements Documents, using agents at the back.
- **Brainstorming** to generate various ideas, apply given constraints. [Example: "Brainstorming with multiple AI AGENTS, Sai Panyam, Innovation Algebra Guest"](https://www.youtube.com/watch?v=82UDm2yVe3Q) Talk
- **Negotiations, Debating** to come at optimized solution
- **Advisory** such as Investment Advisors, which looks at your earnings, risk profile, goals and using multiple specialized expert-agents, comes up with investment plan. [Example](https://github.com/wtlow003/investment-advisor-gpt)
- **Automated Research**: Market Research or even publications/social-media search, collate output, analyze and generate a report.


(Not mission-critical or enterprise level yet, I guess, due to non-definiteness of the output. So, human-in-loop ie in Assistive/Co-pilot mode)


## References

### Learning
- [LangGraph Mastery Playlist - by Sunny Savita](https://www.youtube.com/playlist?list=PLQxDHpeGU14AJ4sBRWLBqjMthxrLXJmgF)
- [AI Agents by Atef Ataya](https://www.youtube.com/playlist?list=PLQog6EfhK_pIVm6A6f-CyZwvZAy5sKmwe)
- [LangGraph: Mastering Advanced AI Workflows by Atef Ataya](https://www.youtube.com/playlist?list=PLQog6EfhK_pJ7I4bLBobe7Yikp5fQfEXU)
- [Large Language Model Agents MOOC, Fall 2024](https://llmagents-learning.org/f24), [Youtube Playlist](https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc)
- [New Course: AI Agents in LangGraph DeepLearningAI](https://www.youtube.com/watch?app=desktop&v=EqEXTGot2xs)
- [Society of Mind](https://en.wikipedia.org/wiki/Society_of_Mind) by Marvin Minsky Intelligence, Graphs, Agents, Memory, [EPub](http://aurellem.org/society-of-mind/)

### QA Agents
<TBD Anjali>

### Companies
- [gpt.ai](https://www.gpt.ai/) AI automations
- [](https://finetune.dev/) by [Julian Saks](https://www.linkedin.com/in/juliansaks/) [Github Memary](https://github.com/kingjulio8238/Memary)  Agent Memories

## Disclaimer

Agents is a research project and should not be used in critical or high-stakes applications without proper testing, evaluation, and risk assessment. The project maintainers and contributors are not responsible for any consequences resulting from the use of this software.

## License

Agents is released under the [MIT License](LICENSE).