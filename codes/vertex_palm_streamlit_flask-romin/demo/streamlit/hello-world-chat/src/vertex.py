

from vertexai.preview.language_models import ChatModel, InputOutputTextPair, ChatSession, TextGenerationModel
import vertexai

PROJECT_ID = "cloud-llm-preview1"
vertexai.init(project=PROJECT_ID, location="us-central1")



def create_session(temperature=0.2,
                   max_output_tokens=256,
                   top_k=40,
                   top_p=.80,
                   context="",
                   examples_for_context=[], 
                   ):
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "context": context,
        "examples": examples_for_context
    }
    # st.write("its using this temperature values for reference = ",temperature)
    # st.write("this is example: ", examples_for_context)
    chat = ChatSession(model=chat_model, **parameters)
    return chat


def response(chat, user_message):
    response = chat.send_message(
        message=user_message
    )
    return response.text

def create_example_InputPutputPair(io_pair):
    example_list = []
    for each_io_pair in ast.literal_eval(io_pair):
        example_list.append(InputOutputTextPair(input_text=each_io_pair['input_text'],
                            output_text=each_io_pair['output_text']
            ))

    return example_list

