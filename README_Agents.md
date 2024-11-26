# Agents

Agents is an open-source project that provides a framework for building and deploying intelligent agents powered by large language models (LLMs). These agents can be used for various tasks such as question-answering, task automation, personalized assistance, and more.

## Introduction

Large Language Models (LLMs) are transformative AI models capable of generating human-like text, understanding natural language, and performing a wide range of tasks. However, deploying and interacting with LLMs can be challenging, especially when it comes to integrating them with other systems and managing their behavior.

LLM-Agents aims to simplify the process of building and deploying LLM-based agents by providing a modular and extensible framework. It abstracts away the complexities of working with LLMs and offers a set of tools and utilities for creating, configuring, and managing intelligent agents.

## Features

- **Agent Creation**: Define agents with specific capabilities, knowledge bases, and personas using a declarative configuration approach, similar to Predibase-Ludwig. Using this config, one can instantiate any agentic framework such as Microsoft's AutoGen or Crew.AI or LangGraph.
- **Modular Design**: Easily swap out or customize different components of an agent, such as the language model, knowledge base, or decision-making strategy.
- **Multi-modal Support**: Agents can process and generate various data types, including text, images, audio, and more.
- **Monitoring and Logging**: Comprehensive logging and monitoring capabilities for tracking agent performance, debugging, and auditing.
- **Deployment Options**: Deploy agents locally, on cloud platforms, or as microservices, with support for scaling and load balancing.
- **Security and Privacy**: Built-in safeguards and controls for ensuring the responsible and ethical use of LLM-based agents.

## Configuration

Say for [this](https://cobusgreyling.medium.com/two-llm-based-autonomous-agents-debate-each-other-e13e0a54429b) agentic systems

```
{
	"agents":
		[
			{
				"name : "AI accelerationist",
				"role": "A passionate visionary who sees the ... ultimately improve society."
			},
			{
				"name : "AI alarmist",
				"role": "A cautious observer of technological advancements. ... security"
			},
		],
		
	"tools":
		[
			{
				"name : "Arxiv",
				"url": "...",
				"key": "..."
			},
			{
				"name : "Duck-Duck-Go Search",
				"url": "...",
				"key": "..."
			},
		],		
	"goal": "The topic for the discussion is 'The current impact of automation and artificial intelligence on employment'.",
	"system_prompt": """
			Your name is {name}.
			Your description is as follows: {description}
			Your goal is to persuade your conversation partner of your point of view.
			DO look up information with your tool to refute your partner’s claims.
			DO cite your sources.
			DO NOT fabricate fake citations.
			DO NOT cite any source that you did not look up.
			Do not add anything else.
			Stop speaking the moment you finish speaking from your perspective.	
	""",
	
```
## Installation

To install LLM-Agents, you'll need Python 3.7 or later and pip. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yogeshhk/Sarvadnya.git
cd src\agents
pip install -r requirements.txt #tbd
```

## Applications
- ** Brainstorming** to generate various ideas, apply given constraints. [Example: "Brainstorming with multiple AI AGENTS, Sai Panyam, Innovation Algebra Guest"](https://www.youtube.com/watch?v=82UDm2yVe3Q) Talk
- ** Negotiations, Debating** to come at optimized solution
- ** Advisory** such as Investment Advisors, which looks at your earnings, risk profile, goals and using multiple specialized expert-agents, comes up with investment plan. [Example](https://github.com/wtlow003/investment-advisor-gpt)


## Contributing

We welcome contributions from the community! If you'd like to contribute to Agents, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive commit messages
4. Push your changes to your forked repository
5. Open a pull request against the main repository

Before submitting a pull request, please ensure that your code follows the project's coding style and that all tests pass.

## References
- [LangGraph Mastery Playlist - by Sunny Savita](https://www.youtube.com/playlist?list=PLQxDHpeGU14AJ4sBRWLBqjMthxrLXJmgF)
- [AI Agents by Atef Ataya](https://www.youtube.com/playlist?list=PLQog6EfhK_pIVm6A6f-CyZwvZAy5sKmwe)
- [LangGraph: Mastering Advanced AI Workflows by Atef Ataya](https://www.youtube.com/playlist?list=PLQog6EfhK_pJ7I4bLBobe7Yikp5fQfEXU)
- [Large Language Model Agents MOOC, Fall 2024](https://llmagents-learning.org/f24), [Youtube Playlist](https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc)
- [New Course: AI Agents in LangGraph DeepLearningAI](https://www.youtube.com/watch?app=desktop&v=EqEXTGot2xs)

Agents builds upon the work of several influential projects and research papers in the field of natural language processing and large language models. Here are some key references:

- [InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Constitutional AI: Harmonic Reinforcement Learning for Artificial General Intelligence](https://arxiv.org/abs/2212.08073)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [https://deepgram.com/learn/llm-agents-when-language-models-do-stuff-for-you](https://deepgram.com/learn/llm-agents-when-language-models-do-stuff-for-you) Good overview

## Disclaimer

Agents is a research project and should not be used in critical or high-stakes applications without proper testing, evaluation, and risk assessment. The project maintainers and contributors are not responsible for any consequences resulting from the use of this software.

## License

Agents is released under the [MIT License](LICENSE).