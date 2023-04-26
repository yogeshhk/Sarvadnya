# Reference paper:
# "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks" https://arxiv.org/abs/1502.05698
# Reference blog:
# https://medium.com/technology-hits/does-chatgpt-really-understand-the-language-4855683b0143

# Assuming OPENAI_API_KEY set in Environment variables

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain
import pandas as pd
bool_score = False
total_score = 0
count = 0

template = "{context} {prompt}"
prompt = PromptTemplate(template=template, input_variables=['context', 'prompt'])

llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

df = pd.read_excel(r'data/Basic20.xlsx')
for index, row in df.iterrows():
    context = (row['Context']).replace("\n", " ")
    prompts = (row['Prompts']).split("\n")
    labels = (row['Labels']).split("\n")
    for prompt, label in zip(prompts, labels):
        print(f"Context: {context}\nPrompt:{prompt}\nLabel: {label}")
        keywords = {'context': context, 'prompt': prompt}
        print(f"Response: {chain.run(keywords).strip()}")
        str_score = input('Score? 0 for Wrong, 1 for Perfect : \n')
        total_score += float(str_score)
        count += 1

print(f"LLM score : {total_score/count}")
