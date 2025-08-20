# # import streamlit as st
# # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # # Load model and tokenizer
# # @st.cache_resource
# # def load_model():
# #     model_dir = "./fine_tuned_model"
# #     tokenizer = AutoTokenizer.from_pretrained(model_dir)
# #     model = AutoModelForCausalLM.from_pretrained(model_dir)
# #     generator = pipeline(
# #         "text-generation",
# #         model=model,
# #         tokenizer=tokenizer
# #     )
# #     return generator

# # generator = load_model()

# # # Streamlit UI
# # st.set_page_config(page_title="Marathi AI Chatbot", page_icon="💬")
# # st.title("💬 Marathi Fine-Tuned Chatbot")

# # # Initialize chat history
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # # Display previous messages
# # for msg in st.session_state.messages:
# #     with st.chat_message(msg["role"]):
# #         st.markdown(msg["content"])

# # # Chat input
# # if prompt := st.chat_input("आपला प्रश्न येथे लिहा..."):
# #     # Store and display user message
# #     st.session_state.messages.append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)

# #     # Generate AI response
# #     with st.chat_message("assistant"):
# #         with st.spinner("विचार करीत आहे..."):
# #             output = generator(prompt, max_length=100, num_return_sequences=1)
# #             response = output[0]["generated_text"]

# #         st.markdown(response)
# #         st.session_state.messages.append({"role": "assistant", "content": response})

# import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Load model and tokenizer
# @st.cache_resource
# def load_model():
#     model_dir = "./fine_tuned_model"
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     model = AutoModelForCausalLM.from_pretrained(model_dir)
#     generator = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         do_sample=False  # deterministic output
#     )
#     return generator

# generator = load_model()

# # Streamlit UI
# st.set_page_config(page_title="Marathi AI Chatbot", page_icon="💬")
# st.title("💬 Marathi Fine-Tuned Chatbot")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Chat input
# if prompt := st.chat_input("आपला प्रश्न येथे लिहा..."):
#     # Store and display user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate AI response
#     with st.chat_message("assistant"):
#         with st.spinner("विचार करीत आहे..."):
#             # Prefix prompt with "Q: ... A:" for clear QA
#             qa_prompt = f"Q: {prompt}\nA:"
            
#             # Extract only the answer text
#             output = generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, top_p=0.9)
#             response = output[0]["generated_text"].strip()
            

#         st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})

# import streamlit as st
# import json

# @st.cache_data
# def load_dataset(uploaded_file):
#     if uploaded_file is not None:
#         # Read JSONL line-by-line
#         lines = uploaded_file.read().decode("utf-8").splitlines()
#         return [json.loads(line) for line in lines]
#     return []

# st.title("Marathi Question Answering (from dataset)")

# uploaded_file = st.file_uploader("Upload your dataset.jsonl", type="jsonl")

# if uploaded_file:
#     dataset = load_dataset(uploaded_file)

#     if not dataset:
#         st.error("Dataset is empty or invalid.")
#         st.stop()

#     # Detect available keys
#     sample_keys = list(dataset[0].keys())
#     st.write("Detected keys in dataset:", sample_keys)

#     # Choose field to search in
#     if "question" in sample_keys:
#         search_field = "question"
#     elif "text" in sample_keys:
#         search_field = "text"
#     else:
#         st.error("Dataset must contain 'question' or 'text' field.")
#         st.stop()

#     # User input
#     search_query = st.text_input("Enter your Marathi question or keyword:")

#     if st.button("Get Answer"):
#         if search_query.strip():
#             query = search_query.strip().lower()

#             results = [
#                 item for item in dataset
#                 if query in item.get(search_field, "").lower()
#             ]

#             if results:
#                 for idx, res in enumerate(results, start=1):
#                     st.markdown(f"**Result {idx}:**")
#                     st.write("**Matched Text:**", res.get(search_field, ""))
#                     st.write("**Answer:**", res.get("answer", "No answer found"))
#             else:
#                 st.error("No matching text found in dataset.")
#         else:
#             st.warning("Please enter a search query.")
import streamlit as st
import json
import re
from pathlib import Path

@st.cache_data
def load_jsonl(uploaded_file):
    return [json.loads(line) for line in uploaded_file]

@st.cache_data
def load_tex_files(uploaded_files):
    data = []
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        # Remove LaTeX commands like \section{}, \textbf{}, etc.
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", content)
        text = re.sub(r"\\[a-zA-Z]+", "", text)  # Remove other commands
        text = re.sub(r"\s+", " ", text).strip()
        data.append({"text": text})
    return data

st.title("Marathi Question Answering (from dataset or .tex files)")

file_type = st.radio("Select file type to upload:", ["JSONL", "TEX"])

if file_type == "JSONL":
    uploaded_file = st.file_uploader("Upload your dataset.jsonl", type="jsonl")
    if uploaded_file:
        dataset = load_jsonl(uploaded_file)
elif file_type == "TEX":
    uploaded_files = st.file_uploader("Upload your .tex files", type="tex", accept_multiple_files=True)
    if uploaded_files:
        dataset = load_tex_files(uploaded_files)

if "dataset" in locals() and dataset:
    sample_keys = list(dataset[0].keys())
    st.write("Detected keys in dataset:", sample_keys)

    # Choose correct key
    if "question" in sample_keys:
        all_items = [item["question"] for item in dataset]
    elif "text" in sample_keys:
        all_items = [item["text"] for item in dataset]
    else:
        st.error("Dataset must contain 'question' or 'text' field.")
        st.stop()

    search_query = st.text_input("Enter your Marathi question or keyword:")

    if st.button("Get Answer"):
        if search_query.strip():
            results_found = False
            for idx, item in enumerate(dataset, start=1):
                text = item.get("text", "")
                # Find sentences containing the search query
                sentences = re.split(r'(?<=[.!?।])\s+', text)
                matches = [s for s in sentences if search_query in s]
                if matches:
                    results_found = True
                    for m in matches:
                        st.markdown(f"**Answer {idx}:** {m.strip()}")
            if not results_found:
                st.error("No matching text found in dataset.")
        else:
            st.warning("Please enter a search query.")
