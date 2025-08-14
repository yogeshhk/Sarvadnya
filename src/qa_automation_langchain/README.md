###################################################################################################
Problem statement : 

Automate QA processes using Agent workflows
1. Agent 1: Create an Agent workflow which will read a requirements document in a doc / txt format   
       Input : Requirements document
       Output : Point wise Summary of Document

2. Agent 2: Based on summarized Requirements document create testcases in a Gherkin format (Given - When - Then)    
       Input : Point wise Summary of Document
       Output : Testcases in Gherkin format
3. Agent 3: Create Selenium testscripts for each scenario in the Gherkin output from Agent 2
       Input : Gherkin testcases
       Output : Selenium automation test scripts 


###################################################################################################

Tech Stack:

1. Groq API
2. LLM model - llama-3.1-8b-instant
3. Agent framework - Langchain
4. UI - Streamlit

###################################################################################################

To execute :

---- streamlit run QAProcessAutomationAgent_UI.py

###################################################################################################

