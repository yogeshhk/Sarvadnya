###################################################################################################
# Problem statement : Create the following workflow using groq API and Langchain
#
# Workflow 1
# Input : Requirements document
# Output : Point wise Summary of Document
#
# Workflow 2
# # Input : Point wise Summary of Document
# # Output : testcases in Gherkin format
#
# Workflow 3
# # Input : Gherkin testcases
# # Output : Selenium testcases for each scenario from Gherkin testcases
###################################################################################################
import requests
import json
import os
from groq import Groq


def simple_AI_Function_Agent(prompt, model="llama-3.3-70b-versatile"):
    try:
        print("Prompt : \n", prompt)

        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )

        response = chat_completion.choices[0].message.content
        return response

    except requests.exceptions.RequestException as e:
        return f"Error connecting to model: {e}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"Error parsing model response: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def convert_requirements_to_testcases(requirements_doc,
                                      workflow_type="Complete Workflow (Summary → Gherkin → Selenium)",
                                      model="llama-3.3-70b-versatile"):
    """
    Convert requirements document to testcases based on the selected workflow

    Args:
        requirements_doc (str): The requirements document text
        workflow_type (str): The type of workflow to run
        model (str): The model to use for generation

    Returns:
        str: The generated output based on the workflow
    """
    result = ""

    if "Generate Summary Only" in workflow_type:
        prompt = "You are an expert in generating QA testcases for any known formats. Generate a concise, point-wise summary of the following requirements document: \n\n" + requirements_doc
        summary = simple_AI_Function_Agent(prompt, model)
        result = f"## Summary of Requirements Document\n\n{summary}"

    elif "Generate Gherkin Testcases Only" in workflow_type:
        prompt = "You are an expert in generating QA testcases in Gherkin formats. Create comprehensive testcases in Gherkin format for the following requirements document: \n\n" + requirements_doc
        gherkin_testcases = simple_AI_Function_Agent(prompt, model)
        result = f"## Gherkin Testcases\n\n{gherkin_testcases}"

    elif "Generate Selenium Testcases Only" in workflow_type:
        prompt = "You are a selenium automation expert. Create Selenium testcases in Python for the following requirements document: \n\n" + requirements_doc
        selenium_testcases = simple_AI_Function_Agent(prompt, model)
        result = f"## Selenium Testcases\n\n{selenium_testcases}"

    else:  # Complete workflow
        # Step 1: Generate summary
        prompt = "You are an expert in generating QA testcases for any known formats. Generate a concise, point-wise summary of the following requirements document: \n\n" + requirements_doc
        summary = simple_AI_Function_Agent(prompt, model)

        # Step 2: Generate Gherkin testcases from summary
        prompt = "You are an expert in generating QA testcases in Gherkin formats. Create comprehensive testcases in Gherkin format using this summary of requirements: \n\n" + summary
        gherkin_testcases = simple_AI_Function_Agent(prompt, model)

        # Step 3: Generate Selenium testcases from Gherkin
        prompt = "You are a selenium automation expert. Create Selenium testcases in Python for each of the following Gherkin scenarios: \n\n" + gherkin_testcases
        selenium_testcases = simple_AI_Function_Agent(prompt, model)

        result = f"## Summary of Requirements Document\n\n{summary}\n\n## Gherkin Testcases\n\n{gherkin_testcases}\n\n## Selenium Testcases\n\n{selenium_testcases}"

    return result


# For standalone execution (will only run if script is executed directly, not imported)
if __name__ == "__main__":
    requirements_doc = """
    The system shall allow users to create and manage their accounts. 
    Users shall be able to log in using their email address and password. 
    The system must provide a password reset functionality. 
    The system shall display a dashboard with key performance indicators. 
    The system shall generate reports in PDF format. 
    The system shall integrate with a third-party payment gateway.
    The system needs to be scalable to handle 10,000 concurrent users.
    The system must ensure data security and comply with GDPR regulations.
    """
    print("#############################################################################")

    prompt = "Generate a summary of following document : " + requirements_doc
    summary = simple_AI_Function_Agent(prompt)
    print("Summary of Requirements document: \n", summary)

    print("#############################################################################")

    prompt = "Create testcases in Gherkin format using summary : " + summary
    gherkin_testcases = simple_AI_Function_Agent(prompt)
    print("Testcases in Gherkin format : \n", gherkin_testcases)

    print("#############################################################################")

    prompt = "Create selenium testcases for each scenario : " + gherkin_testcases
    selenium_testcases = simple_AI_Function_Agent(prompt)
    print(selenium_testcases)