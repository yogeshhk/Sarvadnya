import json
from flask import Flask, render_template, request, jsonify
from vertexai.preview.language_models import ChatModel, InputOutputTextPair, ChatSession, TextGenerationModel
import vertexai
import google.cloud.logging


app = Flask(__name__)
PROJECT_ID = "YOUR_GOOGLE_CLOUD_PROJECT_ID" #Your Google Cloud Project ID
LOCATION = "us-central1"              #us-central1 for now
vertexai.init(project=PROJECT_ID, location=LOCATION)

client = google.cloud.logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "vertex-palm-flask-app-log"
logger = client.logger(log_name)


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
    chat = ChatSession(model=chat_model, **parameters)
    return chat


def response(chat, message):
    response = chat.send_message(
        message=message, max_output_tokens=256, temperature=0.2, top_k=40, top_p=0.8
    )
    return response.text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/palm2', methods=['GET', 'POST'])
def vertexPalM():
    user_input = request.args.get('user_input') if request.method == 'GET' else request.form['user_input']
    logger.log_text(f"Received the following input: {user_input}")
    chat_model = create_session()
    content = response(chat_model,user_input)

    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')