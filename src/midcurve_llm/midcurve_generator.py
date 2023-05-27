# Assuming OPENAI_API_KEY set in Environment variables
import matplotlib.pyplot as plt
from langchain.llms import OpenAI, HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain

prompt = """
You are a geometric transformation program that transforms input 2D polygonal profile to output 1D 
polyline profile. 

Input 2D polygonal profile is defined by set of connected lines with the format as: 
input : [line_1, line_2, line_3,....] where lines are defined by two points, where each point is defined by x and y 
coordinates. So line_1 is defined as ((x_1, y_1), (x_2,y_2)) and similarly the other lines. 

Output is also defined similar to the input as a set of connected lines where lines are defined by two points, 
where each point is defined by x and y coordinates. So, output : [line_1, line_2, line_3,....]

Below are some example transformations, specified as pairs of 'input' and the corresponding 'output'. 

After learning from these examples, predict the 'output' of the last 'input' specified. Do not write code or 
explain the logic but just give the list of lines with coordinates as specified in the 'output' format.

input:[((5.0,5.0), (10.0,5.0)), ((10.0,5.0), (10.0,30.0)), ((10.0,30.0), (35.0,30.0)), ((35.0,30.0), (35.0, 35.0)), 
((35.0, 35.0), (5.0,35.0)), ((5.0,35.0), (5.0,5.0))]
output: [((7.5,5.0), (7.5, 32.5)), ((7.5, 32.5), (35.0, 32.5)), ((35.0, 32.5) (7.5, 32.5))]

input: [((5,5), (10, 5)), ((10, 5), (10, 20)), ((10, 20), (5, 20)), ((5, 20),(5,5))]
output: [((7.5, 5), (7.5, 20))]

input: [((0,25.0), (10.0,25.0)), ((10.0,25.0),(10.0, 45.0)), ((10.0, 45.0),(15.0,45.0)), ((15.0,45.0), (15.0,25.0)), 
((15.0,25.0),(25.0,25.0)), ((25.0,25.0),(25.0,20.0)), ((25.0,20.0),(15.0,20.0)), ((15.0,20.0),(15.0,0)), 
((15.0,0),(10.0,0)), ((10.0,0),(10.0,20.0)), ((10.0,20.0),(0,20.0)), ((0,20.0),(0,25.0))]
output: [((12.5,0), (12.5, 22.5)), ((12.5, 22.5),(12.5,45.0)), ((12.5, 22.5), (0,22.5)), ((12.5, 22.5), (25.0,22.5))]

input:[((0, 25.0), (25.0,25.0)),((25.0,25.0),(25.0,20.0)), ((25.0,20.0),(15.0, 20.0)), ((15.0, 20.0),(15.0,0)), 
((15.0,0),(10.0,0)), ((10.0,0),(10.0,20.0)), ((10.0,20.0),(0,20.0)), ((0,20.0),(0, 25.0))]
output: 

"""

# llms = [{'name': 'Flan', 'model': HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 1e-10})},
#         # {'name': 'OpenAI', 'model': OpenAI(temperature=0)},
#         {'name': 'Bloom', 'model': HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature": 1e-10})}
#         ]
#
# for llm_dict in llms:
#     llm_name = llm_dict['name']
#     llm_model = llm_dict['model']
#     result = llm_model(prompt).strip()
#     llm_dict['result'] = result
#     print(f"model: {llm_name}, result: {result}")



fig, ax = plt.subplots(figsize=(12, 6))


def plot_polyline(lines):
    for line in lines:
        point1 = line[0]
        point2 = line[1]
        xs = [point1[0], point2[0]]
        ys = [point1[1], point2[1]]
        plt.plot(xs, ys)

tshape = [((0, 25.0), (25.0,25.0)),((25.0,25.0),(25.0,20.0)), ((25.0,20.0),(15.0, 20.0)), ((15.0, 20.0),(15.0,0)), ((15.0,0),(10.0,0)), ((10.0,0),(10.0,20.0)), ((10.0,20.0),(0,20.0)), ((0,20.0),(0, 25.0))]
actual = [((12.5,0), (12.5,22.5)), ((12.5,22.5),(25.0,22.5)), ((12.5,22.5),(0,22.5))]
chatgpt = [((2.5, 0), (2.5, 22.5)), ((2.5, 22.5), (2.5, 45.0)), ((2.5, 22.5), (25.0, 22.5)), ((2.5, 22.5), (12.5, 22.5)), ((2.5, 22.5), (0, 22.5)), ((2.5, 22.5), (25.0, 22.5))]
perplexity = [((12.5,0), (12.5, 22.5)), ((12.5, 22.5),(12.5,45.0)), ((12.5, 22.5), (0,22.5)), ((12.5, 22.5), (25.0,22.5))]
bard = [((12.5, 0), (12.5, 25.0)), ((12.5, 25.0), (25.0, 25.0)), ((25.0, 25.0), (25.0, 0))]


plot_polyline(bard)
plt.xlim([-5, 30])
plt.ylim([-5, 30])
plt.show()
