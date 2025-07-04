# Building a GST FAQs App

Goods and Services Tax (GST) in India is complex, and it’s hard for users to find consistent answers. This chatbot simplifies access to GST-related answers using RAG (Retrieval Augmented Generation), enabling natural language queries on structured and unstructured data.

### GST FAQs Chatbot using RAG + LLaMA 3 + Groq

RAG (Retrieval-Augmented Generation) based chatbot that answers Indian GST-related queries using a multi-format knowledge base — structured (CSV) and unstructured (HTML) files.

- Knowledge Base Creation (R): build_QnA_chain() loads GST FAQs from .csv and .html
- Documents are chunked → embedded using MiniLM → indexed using FAISS
- Query Execution (RA(G)): Prompt and context are sent to LLaMA 3 via Groq API to generate a natural, contextual answer

### Commands

```bash
conda activate genai
pip install -r requirements.txt
streamlit run streamlit_main.py
```

Web link : https://askgst-gehcuxfbbasx9enrawjurc.streamlit.app/
Hugging Face Spaces : https://huggingface.co/spaces/chaitraliie/ask-gst

## Demo
Here is a sneak peak of my GST Query Bot, designed to tackle all your GST-related questions effortlessly.
<a href="{https://www.youtube.com/watch?v=K2EmEzVtOZ0}" title="GST FAQ - YouTube"><img src="{https://www.youtube.com/watch?v=K2EmEzVtOZ0}" alt="GST FAQ - YouTube" /></a>

[LinkedIn Post](https://www.linkedin.com/posts/yogeshkulkarni_gst-bot-huggingface-activity-7093395645473972224-O3Y8/)

[Demo Video Aug 2023](./demo_video_aug2023.mp4)
[Demo Video July 2025](./demo_video_jul2025.mp4)
<!-- https://github.com/user-attachments/assets/2d4e2041-4861-4658-a51e-f82beee5b812 -->
