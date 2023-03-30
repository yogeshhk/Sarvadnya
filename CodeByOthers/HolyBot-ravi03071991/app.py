import openai
import pinecone
import gradio as gr
import json
import argparse
import logging

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the console handler
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


class HolyBot:
    def __init__(self):
        try:
            logger.info("Parsing arguments")
            # Parse command-line arguments
            parser = argparse.ArgumentParser()
            parser.add_argument("--holybook", type=str, required=True)
            parser.add_argument("--pinecone_apikey", type=str, required=True)
            parser.add_argument("--pinecone_environment",
                                type=str, required=True)
            parser.add_argument("--openaikey", type=str, required=True)
            args = parser.parse_args()

            self.holybook = args.holybook
            self.pinecone_apikey = args.pinecone_apikey
            self.pinecone_environment = args.pinecone_environment
            self.openaikey = args.openaikey
        except Exception as e:
            logger.error("Error while parsing arguments: {}".format(e))

    # initiates pinecone
    def init_pinecone(self):
        pinecone.init(api_key=self.pinecone_apikey,
                      environment=self.pinecone_environment)

    def qa(self, query):
        # Basic Checks
        if not query:
            return "Please enter your query."

        openai.api_key = self.openaikey

        response = openai.Embedding.create(
            input=[query], model="text-embedding-ada-002")
        embedding = response["data"][0]["embedding"]

        self.init_pinecone()

        index = pinecone.Index(self.holybook)

        with open(f"{self.holybook}.json", "r") as f:
            data = json.loads(f.read())

        res = index.query(vector=(embedding), top_k=8)

        ids = [i["id"] for i in res["matches"]]

        context = ""

        for id in ids:
            context = context + str(id) + ": " + data[str(id)] + "\n\n"

        if self.holybook == "gita":
            book = "Bhagwad Gita"
        else:
            book = self.holybook

        logger.info("Book: {}".format(book))

        systemprompt = f"You are not an AI Language model. You will be a {book} Assistant to the user. Restrict Yourself to the context of the question."

        userprompt = f"Verses:\n\n{context}\n\nQuestion: {query}\n\nAnswer:\n\n"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": userprompt},
            ],
            max_tokens=256,
            temperature=0.0,
        )

        answer = response["choices"][0]["message"]["content"]

        return answer, context

    def cleartext(self, query, output, references):
        """
        Function to clear text
        """
        return ["", "", ""]


if __name__ == "__main__":
    askbook = HolyBot()
    with gr.Blocks() as demo:
        gr.Markdown(
            """
        <h1><center><b>Ask a Holy Book</center></h1>
        """
        )
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(lines=2, label="Enter Your Query.")
                submit_button = gr.Button("Submit")
            with gr.Column():
                ans_output = gr.Textbox(lines=5, label="Answer.")
                references = gr.Textbox(
                    lines=10, label="Relevant Verses.")
                clear_button = gr.Button("Clear")

        # Submit button for submitting query.
        submit_button.click(askbook.qa, inputs=[query], outputs=[
                            ans_output, references])
        # Clear button for clearing query and answer.
        clear_button.click(
            askbook.cleartext,
            inputs=[query, ans_output, references],
            outputs=[query, ans_output, references],
        )
    demo.launch(debug=True)
