# Ref: https://github.com/bhavaniravi/rasa-site-bot
from flask import Flask
from flask import render_template,jsonify,request
from dfengine import DfEngine

app = Flask(__name__)
app.secret_key = '12345'

datafilename = "data/countries.csv"
primarycolumnname = "Country"
dfmodel = DfEngine(datafilename, primarycolumnname)

def get_response(user_message):
    return dfmodel.query(user_message)


@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/chat',methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]
        response_text = get_response(user_message)
        return jsonify({"status":"success","response":response_text})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"Sorry I am not trained to do that yet..."})


app.config["DEBUG"] = True
if __name__ == "__main__":
    app.run(port=8080)