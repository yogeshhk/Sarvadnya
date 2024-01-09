# Ref video CrewAI Tutorial - Next Generation AI Agent Teams (Fully Local) Matthew Berman
# https://www.youtube.com/watch?v=tnejrr-0a94
# https://github.com/joaomdmoura/crewAI
# Using Local LLMs LM Studio Way ############
# https://medium.com/analytics-vidhya/microsoft-autogen-using-open-source-models-97cba96b0f75

import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
import openai
# from langchain.llms import OpenAI
from langchain_openai.llms import OpenAI

# Configure OpenAI settings
# os.environ["OPENAI_API_KEY"] = "YOUR KEY"

lmstudio_llm = OpenAI(temperature=0, openai_api_base="http://localhost:1234/v1")

search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
explorer = Agent(
    role='Senior Researcher',
    goal="""Find and explore the most exciting projects and companies in 
  AI and machine learning in 2024""",
    backstory="""You are and Expert strategist that knows how to spot
              emerging trends and companies in AI, tech and machine learning. 
              You're great at finding interesting, exciting projects in Open 
              Source/Indie Hacker space. You look for exciting, rising
              new github ai projects that get a lot of attention.
              """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=lmstudio_llm
)
writer = Agent(
    role='Senior Technical Writer',
    goal="""Write engaging and interesting blog post about latest AI projects 
            using simple, layman vocabulary""",
    backstory="""You are an Expert Writer on technical innovation, especially 
              in the field of AI and machine learning. You know how to write in 
              engaging, interesting but simple, straightforward and concise. You know 
              how to present complicated technical terms to general audience in a 
              fun way by using layman words.""",
    verbose=True,
    allow_delegation=True,
    llm=lmstudio_llm
)
critic = Agent(
    role='Expert Writing Critic',
    goal="""Provide feedback and criticize blog post drafts. Make sure that the 
            tone and writing style is compelling, simple and concise""",
    backstory="""You are an Expert at providing feedback to the technical
              writers. You can tell when a blog text isn't concise,
              simple or engaging enough. You know how to provide helpful feedback that 
              can improve any text. You know how to make sure that text 
              stays technical and insightful by using layman terms. """,
    verbose=True,
    allow_delegation=True,
    llm=lmstudio_llm
)

# Create tasks for your agents
task_report = Task(
    description="""Make a detailed report on the latest rising projects 
                  in AI and machine learning space. Find emerging trends, topics,
                  rising github projects and producthunt apps that have people talking. 
                  Spot the most recent trends in the last 2-3 weeks.
                  Your final answer MUST be a full analysis report, with bullet points 
                  and with at least 5 exciting new AI projects and tools
                  """,
    agent=explorer
)

task_blog = Task(
    description="""Write a blog article with a short but impactful headline 
                  and at least 10 paragraphs. Blog should summarize the latest 
                  AI and machine learning accomplishments and rising github projects and 
                  companies in the AI space. Style and tone should be compelling and concise,
                  fun, technical but also use layman words for the general public. Name 
                  specific new, exciting projects, apps and companies in AI world.
                  """,
    agent=writer
)

task_critique = Task(
    description="""Identify parts of the blog that aren't written concise enough
                   and improve them. Make sure that the blog has engaging 
                  headline with 30 characters max, and that there are at least 10 paragraphs. 
                  Blog needs to be written in such a way that a 15-year old 
                  can  understand it.
                  """,
    agent=critic
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=2,
    process=Process.sequential  # Sequential process will have tasks executed one after the other
    # and the outcome of the previous one is passed as extra content into this next.
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
