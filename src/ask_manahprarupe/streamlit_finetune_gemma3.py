# import streamlit as st
# import torch
# from transformers import AutoTokenizer, pipeline
# from peft import AutoPeftModelForCausalLM
# import re

# # ======================
# # 1. Load Model + Tokenizer
# # ======================
# MODEL_PATH = "./gemma3-marathi-lora"  # your fine-tuned LoRA folder

# @st.cache_resource
# def load_model():
#     model = AutoPeftModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.float32,
#         device_map=None  # CPU only
#     )
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         torch_dtype=torch.float32,
#         device=-1
#     )
#     return pipe, tokenizer

# pipe, tokenizer = load_model()

# # ======================
# # 2. Streamlit UI
# # ======================
# st.set_page_config(page_title="Marathi Chatbot", page_icon="🪔", layout="centered")
# st.title("🪔 Marathi Chatbot (Gemma3 LoRA)")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display conversation
# for msg in st.session_state.messages:
#     role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
#     st.markdown(f"**{role}:** {msg['content']}")

# # Input box
# user_input = st.text_area("✍️ तुमचा प्रश्न लिहा:", "")

# # Handle button click
# if st.button("उत्तर मिळवा"):
#     if user_input.strip():
#         # Save user message
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         with st.spinner("🤔 विचार चालू आहे..."):
#             # Instruction to force Marathi output
#             final_prompt = (
#                 "तुम्ही एक हुशार सहाय्यक आहात जो नेहमी मराठीत, सोप्या आणि स्पष्ट भाषेत उत्तर देतो.\n\n"
#                 "उदाहरण:\n"
#                 "प्रश्न: Lindy Effect म्हणजे काय?\n"
#                 "उत्तर: लिंडी इफेक्ट म्हणजे एखादी गोष्ट जितकी जुनी असेल तितकी ती पुढे टिकण्याची शक्यता जास्त असते.\n\n"
#                 f"प्रश्न: {user_input}\nउत्तर:"
#             )


#             output = pipe(
#                 final_prompt,
#                 max_new_tokens=200,
#                 do_sample=True,
#                 temperature=0.6,   # slightly lower = more focused
#                 top_p=0.85,
#                 repetition_penalty=1.5,
#                 eos_token_id=tokenizer.eos_token_id
#             )

#             # Extract and clean response
#             response = output[0]["generated_text"].replace(final_prompt, "").strip()
#             response = re.sub(r"<.*?>", "", response)   # remove HTML tags
#             response = re.sub(r"[a-zA-Z]+", "", response)  # drop stray English
#             response = re.sub(r"\s+", " ", response).strip()  # clean spaces

#         # Save bot message
#         st.session_state.messages.append({"role": "bot", "content": response})

#         # Refresh chat
#         st.rerun()

#//generatting random ques

# import streamlit as st
# import torch
# from transformers import AutoTokenizer, pipeline
# from peft import AutoPeftModelForCausalLM
# import re

# # ======================
# # 1. Load Model + Tokenizer
# # ======================
# MODEL_PATH = "./gemma3-marathi-lora"  # your fine-tuned LoRA folder

# @st.cache_resource
# def load_model():
#     model = AutoPeftModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.float32,
#         device_map=None  # CPU only
#     )
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         torch_dtype=torch.float32,
#         device=-1
#     )
#     return pipe, tokenizer

# pipe, tokenizer = load_model()

# # ======================
# # 2. Streamlit UI
# # ======================
# st.set_page_config(page_title="Marathi Chatbot", page_icon="🪔", layout="centered")
# st.title("Marathi Chatbot (Gemma3 LoRA)")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display conversation
# for msg in st.session_state.messages:
#     role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
#     st.markdown(f"**{role}:** {msg['content']}")

# # Input box
# user_input = st.text_area("✍️ तुमचा प्रश्न लिहा:", "")

# # Handle button click
# if st.button("उत्तर मिळवा"):
#     if user_input.strip():
#         # Save user message
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         with st.spinner("🤔 विचार चालू आहे..."):
#             # Match the fine-tuned training format
#             final_prompt = (
#                 f"तुम्ही एक उपयुक्त सहाय्यक आहात जो नेहमी मराठीत, थोडक्यात आणि स्पष्टपणे उत्तर देतो.\n\n"
#                 f"प्रश्न: {user_input}\n"
#                 f"उत्तर (थोडक्यात, स्पष्ट अर्थ):"
#             )

#             output = pipe(
#                 final_prompt,
#                 max_new_tokens=200,
#                 do_sample=True,
#                 temperature=0.6,   # lower = more focused
#                 top_p=0.9,
#                 repetition_penalty=1.3,
#                 eos_token_id=tokenizer.eos_token_id
#             )

#             # Extract and clean response
#             generated_text = output[0]["generated_text"]
#             response = generated_text[len(final_prompt):].strip()  # cut prompt part
#             response = re.sub(r"<.*?>", "", response)  # remove HTML tags
#             response = re.sub(r"\s+", " ", response).strip()  # clean spaces

#         # Save bot message
#         st.session_state.messages.append({"role": "bot", "content": response})

#         # Refresh chat
#         st.rerun()

#upoad file
# import streamlit as st
# import torch
# from transformers import AutoTokenizer, pipeline
# from peft import AutoPeftModelForCausalLM
# import re

# MODEL_PATH = "./gemma3-marathi-lora"  # your fine-tuned LoRA

# @st.cache_resource
# def load_model():
#     model = AutoPeftModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.float32,
#         device_map=None
#     )
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         torch_dtype=torch.float32,
#         device=-1  # CPU
#     )
#     return pipe, tokenizer

# pipe, tokenizer = load_model()

# st.title("📖 Marathi Q&A from .tex files")

# # Upload .tex
# uploaded_file = st.file_uploader("Upload .tex file", type=["tex"])

# def clean_tex(tex_text):
#     text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", tex_text)
#     text = re.sub(r"\\[a-zA-Z]+", "", text)
#     text = re.sub(r"\{|\}", "", text)
#     return text.strip()

# context = ""
# if uploaded_file:
#     tex_text = uploaded_file.read().decode("utf-8")
#     context = clean_tex(tex_text)
#     st.text_area("Extracted Marathi Text", context, height=250)

# # Ask Question
# question = st.text_input("तुमचा प्रश्न विचारा (in Marathi):")

# if question and context:
#     with st.spinner("🤔 विचार चालू आहे..."):
#         final_prompt = (
#             "तुम्ही एक हुशार सहाय्यक आहात जो नेहमी मराठीत, स्पष्ट भाषेत उत्तर देतो.\n\n"
#             f"संदर्भ मजकूर:\n{context}\n\n"
#             f"प्रश्न: {question}\nउत्तर:"
#         )

#         output = pipe(
#             final_prompt,
#             max_new_tokens=200,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9,
#             repetition_penalty=1.3,
#             eos_token_id=tokenizer.eos_token_id
#         )

#         response = output[0]["generated_text"].replace(final_prompt, "").strip()
#         response = re.sub(r"<.*?>", "", response)
#         response = re.sub(r"\s+", " ", response).strip()

#     st.success("📖 उत्तर:")
#     st.write(response)
import streamlit as st
import torch
from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM
import re

MODEL_PATH = "./gemma3-marathi-lora"  # your fine-tuned LoRA

@st.cache_resource
def load_model():
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32
    )
    return pipe, tokenizer

pipe, tokenizer = load_model()

st.title("📖 Marathi Q&A from .tex files")

# Upload .tex
uploaded_file = st.file_uploader("Upload .tex file", type=["tex"])

def clean_tex(tex_text):
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", tex_text)   # remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"\{|\}", "", text)
    return text.strip()

context = ""
if uploaded_file:
    tex_text = uploaded_file.read().decode("utf-8")
    context = clean_tex(tex_text)
    st.text_area("Extracted Marathi Text", context, height=250)

# Ask Question
question = st.text_input("तुमचा प्रश्न विचारा (in Marathi):")

if question and context:
    with st.spinner("🤔 विचार चालू आहे..."):
        # Force Marathi output by explicitly instructing the model
        final_prompt = (
            f"फक्त मराठीत उत्तर द्या.\n"
            f"खालील संदर्भावर आधारित प्रश्नाचे उत्तर मराठीत द्या.\n\n"
            f"संदर्भ:\n{context}\n\n"
            f"प्रश्न: {question}\n\n"
            f"उत्तर:"
        )

        output = pipe(
            final_prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

        response = output[0]["generated_text"]
        # Remove the prompt part
        response = response.replace(final_prompt, "").strip()
        # Remove any unwanted tokens or HTML
        response = re.sub(r"<.*?>", "", response)
        response = re.sub(r"\s+", " ", response).strip()

    st.success("📖 उत्तर:")
    st.write(response)
