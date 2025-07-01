import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from colorama import Fore, init


load_dotenv()
init(autoreset=True)

def ask_inputs():
    edu_phases = {
        0: "> 18 (School)",
        1: "18–21 (College)",
        2: "22–30 (Early Job)",
        3: "30–50 (Mid Career)",
        4: "50–60 (Giving back?)"
    }

    print(Fore.MAGENTA + "Choose career phase:")
    for i, ph in edu_phases.items():
        print(f"[{i}] {ph}")
    phase_input = int(input("=> "))
    phase = f"If my current career phase is shown as 'Age range (phase)', then I am in '{edu_phases[phase_input]}' of my life."

    domains = input(Fore.YELLOW + "\nWhat are your current domains (e.g., project management, testing, etc)?\n=> ")

    ratings = {0: "No", 1: "Can try", 2: "Absolutely"}
    expertise = {0: "a novice", 1: "an intermediate", 2: "an expert"}

    def ask_rating(question, color):
        print(color + "\n" + question)
        for i, label in ratings.items():
            print(f"[{i}] {label}")
        choice = int(input("=> "))
        return expertise[choice]

    maths = "In mathematics, I am " + ask_rating("Can you write gradient descent equation in two variables?", Fore.RED)
    programming = "In programming, I am " + ask_rating("Can you code matrix multiplication?", Fore.BLUE)
    ml = "In machine learning, I am " + ask_rating("Can you explain a Confusion Matrix?", Fore.CYAN)

    prep = int(input(Fore.GREEN + "\nIn how many months do you want to switch to data science?\n=> "))
    return phase, domains, maths, programming, ml, prep

def main():
   
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing TOGETHER_API_KEY in .env file")

    phase, domains, maths, programming, ml, prep = ask_inputs()

    prompt_str = (
        "You are an expert career counsellor specializing in guiding career transitions into data science.\n"
        "{phase}\n"
        "So far, I have been working in the domain(s) of {domains}.\n"
        "{maths}\n"
        "{programming}\n"
        "{ml}\n"
        "I wish to change my career to data science in the next {prep} months.\n"
        "Based on the above background, suggest a detailed, month-wise preparation plan.\n"
        "Include what articles to read, YouTube videos to watch, courses to take, certifications to pursue, etc.\n\n"
        "Plan:\n"
    )

    prompt_template = PromptTemplate(
        template=prompt_str,
        input_variables=["phase", "domains", "maths", "programming", "ml", "prep"]
    )

    
    llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_key=api_key,
    base_url="https://api.together.xyz/v1" #this url is commonly used for Together AI models
)

    chain: RunnableSequence = prompt_template | llm

    print(Fore.CYAN + "\n==================== Together AI Response ====================")

    response = chain.invoke({
        "phase": phase,
        "domains": domains,
        "maths": maths,
        "programming": programming,
        "ml": ml,
        "prep": prep
    })

    print(Fore.GREEN + response.content)
    print(Fore.CYAN + "=============================================================")

if __name__ == "__main__":
    main()
