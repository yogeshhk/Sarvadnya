# import os
# from llama_index.core import (
#     Settings, StorageContext, VectorStoreIndex,
#     load_index_from_storage
# )
# from llama_index.core.readers import SimpleDirectoryReader
# from llama_index.embeddings.langchain import LangchainEmbedding
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from rich.console import Console
# from rich.text import Text
# from rich.panel import Panel

# # Configuration
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# DATA_DIR = "C:/Users/OMEN/OneDrive/Desktop/Warmup/Sarvadnya_internship/src/ask_kautilya/data/"
# INDEX_DIR = "C:/Users/OMEN/OneDrive/Desktop/Warmup/Sarvadnya_internship/src/ask_kautilya/model"

# def set_local_embedding():
#     Settings.llm = None
#     Settings.embed_model = LangchainEmbedding(
#         HuggingFaceEmbeddings(
#             model_name=MODEL_NAME,
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True}
#         )
#     )

# def construct_index(directory_path: str):
#     set_local_embedding()
#     documents = SimpleDirectoryReader(directory_path).load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     index.storage_context.persist(persist_dir=INDEX_DIR)
#     print("✅ Index built and saved at:", INDEX_DIR)

# # def interactive_chat():
# #     set_local_embedding()  # ✅ Must set again here!
# #     storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
# #     index = load_index_from_storage(storage)
# #     query_engine = index.as_query_engine()
# #     print("\n🤖 Ask Kautilya is ready! Type 'exit' to quit.\n")

# #     while True:
# #         q = input("You: ")
# #         if q.strip().lower() in ["exit", "quit"]:
# #             print("👋 Goodbye!")
# #             break
# #         response = query_engine.query(q)
# #         print("\nBot:", response, "\n")
# console = Console()

# def interactive_chat():
#     set_local_embedding()  # Must reapply embedding settings
#     storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
#     index = load_index_from_storage(storage)
#     query_engine = index.as_query_engine()

#     console.print(Panel.fit("🤖 [bold cyan]Ask Kautilya is Ready![/bold cyan]\n[dim]Type 'exit' to quit[/dim]", title="Welcome", subtitle="Arthashastra Bot"))

#     while True:
#         q = console.input("[bold yellow]You:[/bold yellow] ")

#         if q.strip().lower() in ["exit", "quit"]:
#             console.print("\n[bold red]👋 Goodbye![/bold red]")
#             break

#         response = query_engine.query(q)

#         # Format the bot's response in a pretty panel
#         console.print(Panel.fit(str(response), title="[green]Bot[/green]", border_style="cyan"))

# if __name__ == "__main__":
#     if not os.path.isdir(INDEX_DIR) or not os.path.exists(os.path.join(INDEX_DIR, "docstore.json")):
#         print("🚨 Index not found. Building it now...")
#         construct_index(DATA_DIR)

#     interactive_chat()
import os
from llama_index.core import (
    Settings, StorageContext, VectorStoreIndex, load_index_from_storage
)
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings

from rich.console import Console
from rich.panel import Panel

# ——— Configuration ———
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = (
    "C:/Users/OMEN/OneDrive/Desktop/Warmup/"
    "Sarvadnya_internship/src/ask_kautilya/data/"
)
INDEX_DIR = (
    "C:/Users/OMEN/OneDrive/Desktop/Warmup/"
    "Sarvadnya_internship/src/ask_kautilya/model/"
)

console = Console()


def set_local_embedding():
    """Use a local HuggingFace model for embeddings."""
    Settings.llm = None
    Settings.embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def construct_index(directory_path: str):
    """Build and persist index from the given data directory."""
    set_local_embedding()
    documents = SimpleDirectoryReader(
    directory_path,
    chunk_size=512,
    chunk_overlap=50
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    console.print(f"[green]✅ Index built and saved at:[/green] {INDEX_DIR}")


def interactive_chat():
    """Start an interactive Rich-powered console chat."""
    set_local_embedding()
    storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage)

    # Use compact mode to reduce long irrelevant dumps
    query_engine = index.as_query_engine(response_mode="compact", similarity_top_k=2)

    console.print(
        Panel.fit(
            "🤖 [bold cyan]Ask Kautilya is Ready![/bold cyan]\n[dim]Type 'exit' to quit[/dim]",
            title="Welcome",
            border_style="cyan",
        )
    )

    while True:
        q = console.input("[bold yellow]You:[/bold yellow] ").strip()
        if q.lower() in {"exit", "quit"}:
            console.print("[bold red]👋 Goodbye![/bold red]")
            break

        try:
            response = query_engine.query(q)
            # Get only the clean answer from the response object
            answer = getattr(response, "response", str(response)).strip()
            answer = answer[:1000] + "..." if len(answer) > 1000 else answer

            console.print(
                Panel(
                    answer,
                    title="[green]Bot[/green]",
                    border_style="magenta",
                    expand=False,
                )
            )
        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] {e}")


if __name__ == "__main__":
    if not os.path.isdir(INDEX_DIR) or not os.path.exists(os.path.join(INDEX_DIR, "docstore.json")):
        console.print("[yellow]🚨 Index not found. Building it now...[/yellow]")
        construct_index(DATA_DIR)

    interactive_chat()
