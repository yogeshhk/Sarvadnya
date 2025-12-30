# Prompts Used

## Building APP

You are an expert in building generative AI apps. Please write an application to create a chatbot on data files having descriptions of mental models written in marathi. Please prepare 5 files: app.py will have UI code in streamlit. rag.py will have a class encapsulating retrieval augmented generation functionality using llamaindex. It will index data files in local chroma db, use gemma model via Groq APIs. and after retrieval, will have prompt to validate the answers. fine-tune.py will have class using unsloth code to fine-tune raw gemma model using LORA using the data files.

UI will call member functions of respective classes.

UI should have drop-down to chose between raw gemma model or fine-tuned gemma model.

In __main__ section of rag.py and fine-tune.py have a few testing code to test respective code within that file only, especially inference tests. Like ask question in Marathi "Sunk cost fallacy ya mental models la marathi t kaay mhanatat ani tyache udaharan dyaa"

write an essential README file with title Ask Vichar-Chitre chatbot, for this GitHub repository. Also write requirements.txt for list of installations to be done.

## Generating Questions Answers Evaluation pair
You are an expert in generating question answers pair from Marathi blogs.
For the attached blogs, please generate about 5 distinct questions and their corresponding answers, for each blog. Answers should be within the blog only, preferably a one liner. Output should have 3 columns in "|" separated manner. First column is for the questions, second column is for the answers and the third column is for the location of the answer, format: filename: line number

## Writing Evaluation Framework
You are an expert in evaluating RAG output by different evaluation metrics, especially'LLM as a judge'. Please write file 'rag_evaluation.py' which will import attached 'evaluation_set.csv' which contains questions, correct answers and location of those answers. Write an evluation class. Should load the csv, take argument like LLM model name, which defaulted to gemma like below.

groq_api_key = os.getenv("GROQ_API_KEY")
LLM_MODEL_AT_GROQ = "llama-3.1-8b-instant"
EMBEDDING_MODEL_AT_HUGGINGFACE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

Similar to the way RAGChatbot has been used to get answers for gen questions. let the evaluation class get predicted answers from all the questions in the csv, then compare the predicted answers with the correct answers from the csv using known NLP methods such as BLEU, ROUGE, and METEOR and embedding based semantic similarity, plus 'LLM as a judge'. Compare predicted context location with the correct location also. Based on the matching score, calculate the cumulative accuracy.

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY missing. Set it in .env")
    else:
		question = "Inversion म्हणजे काय?"
        bot = RAGChatbot(data_directory="data", groq_api_key=groq_api_key)
        result = bot.get_response(question)
        print("\n🧠 उत्तर:", result["answer"])
        print("\n📎 संदर्भ:\n", result["context"][:500])