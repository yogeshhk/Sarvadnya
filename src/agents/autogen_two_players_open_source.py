# https://github.com/microsoft/autogen/blob/osllm/notebook/open_source_language_model_example.ipynb

# >> modelz-llm -m bigscience/bloomz-560m --device auto [NOT FOR WINDOWS]
# >> python -m llama_cpp.server --model <model path>.gguf

# Setup autogen with the correct API
import autogen
from autogen import AssistantAgent, UserProxyAgent

autogen.oai.ChatCompletion.start_logging()

local_config_list = [
        {
            'model': 'llama-7b.ggmlv3.q4_0.gguf.bin',
            'api_key': 'any string here is fine',
            'api_type': 'openai',
            'api_base': "http://localhost:8000",
            'api_version': '2023-03-15-preview'
        }
]

# # Perform Completion
# question = "Who are you?"
# response = autogen.oai.Completion.create(config_list=local_config_list, prompt=question, temperature=0)
# ans = autogen.oai.Completion.extract_text(response)[0]
#
# print("The small model's answer is:", ans)

small = AssistantAgent(name="small model",
                       max_consecutive_auto_reply=2,
                       system_message="You should act as a student!",
                       llm_config={
                           "config_list": local_config_list,
                           "temperature": 1,
                       })

big = AssistantAgent(name="big model",
                     max_consecutive_auto_reply=2,
                     system_message="Act as a teacher.",
                     llm_config={
                         "config_list": local_config_list,
                         "temperature": 1,
                     })

big.initiate_chat(small, message="Who are you?")
