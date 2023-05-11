# 
# Assuming OPENAI_API_KEY set in Environment variables

from langchain.llms import OpenAI, HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain
from colorama import Fore


def ask_inputs():
    edu_phases = {0: "> 18 (School)", 1: "18-21 (College)", 2: "22-30 (Early Job)", 3: "30-50(Mid Career)",
                  4: "50-60(Giving back?)"}
    edu_phases_str = " choose career phase: age (phase): "
    for i, ph in edu_phases.items():
        edu_phases_str += f"[{i}] {ph}\n"
    edu_phases_str += "\n=> "
    phase_input = int(input(Fore.MAGENTA + edu_phases_str))
    phase = "If my current career phase is shown in the format as 'Age range in years (phase)' then I am in '" + \
            edu_phases[phase_input] + "' of my life."
    domains = input(Fore.YELLOW + "What are your current domains, such as project management, testing, etc: ")

    ratings = {0: "No", 1: "Can try", 2: "Absolutely"}
    expertize = {0: "a novice", 1: "an intermediate", 2: "an expert"}

    ratings_str = "\nChoose level: \n"
    for i, rt in ratings.items():
        ratings_str += f"[{i}] {rt}\n"

    math_question = "Can you write gradient descent equation in two variables? "
    maths_input = int(input(Fore.RED + math_question + ratings_str + "\n=> "))
    maths = "In mathematics, I am " + expertize[maths_input]

    programming_question = "Can you code matrix multiplication, now? "
    programming_input = int(input(Fore.BLUE + programming_question + ratings_str + "\n=> "))
    programming = "In programming, I am " + expertize[programming_input]

    machinelearning_question = "Can you explain Confusion Matrix? "
    ml_input = int(input(Fore.MAGENTA + machinelearning_question + ratings_str + "\n=> "))
    ml = "In machine learning, I am " + expertize[ml_input]

    prep = int(input(Fore.YELLOW + "In how many months you have to switch "))
    return phase, domains, maths, programming, ml, prep


def main():
    phase, domains, maths, programming, ml, prep = ask_inputs()

    prompt_str = "You are an expert career counsellor specializing in guiding career transitions to data science. " \
                 "{phase}. So far I have been working in domains of {domains}. {maths}. {programming}. {ml}. I wish " \
                 "to change my career to data science in coming {prep} months, so I have only that much time to " \
                 "prepare. With the above background suggest a detailed month-wise plan for preparation, including " \
                 "articles to read, YouTube videos to watch, courses to take, certifications to do, etc. \n Plan: \n"

    prompt_template = PromptTemplate(template=prompt_str, input_variables=['phase', 'domains', 'maths',
                                                                           'programming', 'ml', 'prep'])

    # prompt_template.format()
    models_list = [
        # {'name': 'Vicuna', 'model': HuggingFaceHub(repo_id="jeffwan/vicuna-13b")},
        {'name': 'OpenAI', 'model': OpenAI(temperature=0)}]

    for llm_dict in models_list:
        print(Fore.CYAN + "===========================")
        llm_name = llm_dict['name']
        print(Fore.RED + llm_name)
        llm_model = llm_dict['model']
        chain = LLMChain(llm=llm_model, prompt=prompt_template, verbose=False)
        response = chain.run(phase=phase, domains=domains, maths=maths, programming=programming, ml=ml, prep=prep)
        print(Fore.GREEN + response)
        print(Fore.CYAN + "===========================")


if __name__ == "__main__":
    main()
