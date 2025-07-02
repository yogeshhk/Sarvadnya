# streamlitfalcon.py
import streamlit as st
from falcon import generate_response, get_text

st.set_page_config(page_title="Groq Falcon Chat")
st.title(" Falcon-like Chatbot (via Groq LLaMA3)")

if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []

user_input = get_text()
if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state.generated:
    st.markdown("---")
    for user, bot in zip(st.session_state.past, st.session_state.generated):
        st.markdown(f"**You**: {user}")
        st.markdown(f"**Falcon**: {bot}")
        st.markdown("----")
