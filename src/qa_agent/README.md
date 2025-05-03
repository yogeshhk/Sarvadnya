# **AI-Powered Testcase Generation Agent**  

Large Language Models (LLMs) have gained popularity as coding assistants, but their potential extends beyond development—into testing as well. By leveraging LLMs to generate test cases directly from requirement documents, we can ensure that development is driven not just by functionality but also by passing all necessary test scenarios.

This repository presents a small Proof of Concept (PoC) demonstrating how LLMs can generate test cases in different formats based on a given requirement document. To maintain control over workflow while still utilizing LLM capabilities, this project integrates Langraph, which structures LLM-driven decision-making within a predetermined sequence.

This repository contains an AI-driven agent workflow that processes a requirements document, summarizes it, and generates test cases in a selected format. It leverages **GROQ - Llama-3.2 1B-instruct** for AI processing and **LangGraph framework** for agent orchestration.  

## How It Works  

This PoC follows a structured workflow to generate test cases from requirement documents. The process consists of the following steps:  

1. **Open the Requirement Document**  
   - The system loads a requirement document provided by the user.  
   - If no document is selected, it defaults to using `content.txt`.  

2. **Select an LLM**  
   - Users can choose from available open-source language models to generate test cases.  
   - If no selection is made, the default model is **LLaMA 3**.  

3. **Choose Test Case Format**  
   - The system asks the user which format the test cases should be generated in.  
   - Currently supported formats:  
     - **Gherkin** (for behavior-driven development)  
     - **Selenium** (for automated web testing)  

4. **Generate and Output Test Cases**  
   - The system uses the selected LLM to generate detailed test cases in the chosen format.  
   - At present, the generated test cases are verbose and descriptive.  
   - With further prompt optimization, the outputs can be refined into directly runnable test scripts where applicable.  

This workflow ensures that test generation is structured while still leveraging the power of LLMs at various stages, preventing the unpredictability often associated with autonomous AI-driven decision-making.

## **Problem Statement**  

The goal of this project is to develop an AI-powered agent workflow that automates the generation of test cases from a given requirements document. The process consists of the following steps:  

1. **Summarizing the Requirements Document**  
   - **Agent 1** reads the requirements document (in `.doc` or `.txt` format) and extracts a point-wise summary.  
   - **Input:** Requirements document  
   - **Output:** Point-wise summary of the document  

2. **Router Functionality**  
   - The summarized requirements are formatted based on the desired test case structure (e.g., Gherkin, Selenium, etc.).  

3. **Generating Test Cases**  
   - **Agent 2** takes the summarized requirements and generates test cases in the selected format.  
   - **Input:** Point-wise summary of the document  
   - **Output:** Test cases in the selected format  

## **Tech Stack**  

- **AI Model:** GROQ - Llama-3.2 1B-instruct  
- **Agent Orchestration:** LangGraph framework  

## **Installation & Usage**  

### **1. Clone the Repository**  
```sh
git clone https://github.com/yogeshhk/MicroSaas.git
```

### **2. Navigate to the Project Directory**  
```sh
cd MicroSaaS/src/qa_agent
```

### **3. Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4. Set Up API Keys**  
Create a `.streamlit` folder in the root directory and add a `secrets.toml` file with the following contents:  
```toml
GROQ_API_KEY = "your_GROQ_api_key"
TAVILY_API_KEY = "your_Tavily_api_key"
```

### **5. Run the Application**  
```sh
python -m streamlit run app.py
```

## **Contributing**  

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.  

## **License**  

This project is licensed under the **MIT License**.  
