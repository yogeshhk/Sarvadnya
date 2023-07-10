import streamlit as st
from vertexai.preview.language_models import ChatModel, InputOutputTextPair, ChatSession, TextGenerationModel
import vertexai
from PIL import Image
from src.utils import *
from src.vertex import *
from streamlit_chat import message



st.set_page_config(
    page_title="Vertex PaLM Chat API",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This app shows you how to use Vertex PaLM Chat API"
    }
)


image = Image.open('/Users/lavinigam/Documents/office-work/LLM/vertex_palm_streamlit_flask/demo/streamlit/hello-world-chat/image/palm.jpg')
st.image(image)
st.title(":sunglasses: :red[PaLM 2] :blue[Vertex AI] Chat :orange[Demo] :sunglasses: :white_check_mark: ")
# st.markdown("<h1 style='text-align: center; color: black;'>PaLM 2 Vertex AI Chat Demo</h1>", unsafe_allow_html=True)

#creating session states    
create_session_state()


#defining tabs 
# setting_tab, chat_tab = st.columns(2)

with st.sidebar:
    image = Image.open('/Users/lavinigam/Documents/office-work/LLM/vertex_palm_streamlit_flask/demo/streamlit/hello-world-chat/image/sidebar_image.jpg')
    st.image(image)
    st.markdown("<h2 style='text-align: center; color: red;'>Setting Tab</h1>", unsafe_allow_html=True)


    st.write("Model Settings:")

    #define the temeperature for the model 
    temperature_value = st.slider('set temperature :', 0.0, 1.0, 0.2)
    st.session_state['temperature'].append(temperature_value)

    #define the context and profil to the model. 
    context_value = st.text_area("Add custom behaviour/profile to your bot: ",
                                    placeholder = "Be creative and write the persona you want the bot to take and define explicitly about its expertise.")
    st.session_state['context'] = context_value

    #define the examples given to the model. 
    example_value = st.text_area("Add examples that shows the bot how to respond: ")
    if example_value:
        st.session_state['example'] = create_example_InputPutputPair(example_value)

    if not st.session_state['chat_model']:
        if st.button("Create your chat session"):
            chat_model = create_session(temperature = temperature_value,
                                                context = context_value,
                                                examples_for_context= st.session_state['example']
                                                )
            st.session_state['chat_model'] = chat_model
            st.write("You are now connected to PaLM 2 and ready to chat....")
    else:
        st.write("You are connected to PaLM 2.........")
    #define some chat setting.
    st.write("Chat Settings:")
    if st.button("Clear Chat"):
        clear_chat()
    # if st.button("Hard Reset - Be careful. Not stable."):
    #     hard_reset_session()
    debug_mode_choice = st.radio("debug mode ", (False,True))
    st.session_state['debug_mode'] = debug_mode_choice
    if st.button("Download Chat"):
        download_chat()


with st.container():
# with setting_tab:
    st.write("Current Bot Settings: ")
    if st.session_state['temperature'] or st.session_state['debug_mode'] or st.session_state['context']:
        st.write ("Temperature: ",st.session_state['temperature'][-1]," \t \t Debug Model: ",st.session_state['debug_mode'])
        st.write ("Context: ",st.session_state['context'])

with st.container():
    user_input = st.text_input('Your message to the bot:', key='chat_widget', on_change=chat_input_submit)

    if st.session_state.chat_input:
        #call the vertex PaLM API and send the user input
        with st.spinner('PaLM is working to respond back, wait.....'):
            
            try:
                bot_message = response(st.session_state['chat_model'], st.session_state.chat_input)
            
                #store the output
                if len(st.session_state['past'])>0:
                    if st.session_state['past'][-1] != st.session_state.chat_input:
                        st.session_state['past'].append(st.session_state.chat_input)
                        st.session_state['generated'].append(bot_message)
                else:
                    st.session_state['past'].append(st.session_state.chat_input)
                    st.session_state['generated'].append(bot_message)

            except AttributeError:
                st.write("You have not created the chat session. On left sidebar, click on 'Create your chat session'")

    #display generated response 
    if st.session_state['generated'] and st.session_state['past']:
        for i in range(len(st.session_state["generated"])-1,-1,-1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='big-smile')
            message(st.session_state["generated"][i], key=str(i), avatar_style='bottts')

    if st.session_state['debug_mode']:
        st.write("len of generated response: ",len(st.session_state["generated"]))
        st.write(f'Last mssage to bot: {st.session_state.chat_input}')
        st.write(st.session_state)


