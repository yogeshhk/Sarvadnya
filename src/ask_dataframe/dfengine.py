import pandas as pd
import random
import requests


class DfEngine:
    def __init__(self, datafilename, primarycolumnname):
        self.datafilename = datafilename
        self.primarycolumnname = primarycolumnname
        self.df = pd.read_csv(self.datafilename, encoding="ISO-8859-1")
        self.primary_key_values_list = self.df[self.primarycolumnname].tolist()
        self.std_responses = {
            'greet': ["Hello!", "Hi there!", "Hey! How can I help you?"],
            'affirm': ["Glad to hear that!", "Great!", "Awesome!"],
            'goodbye': ["Bye!", "See you later!", "Goodbye!"]
        }

    def query_rasa(self, message):
        """Send user message to local Rasa HTTP server and return parsed intent/entities"""
        try:
            response = requests.post(
                "http://localhost:5005/model/parse",
                json={"text": message}
            )
            return response.json()
        except Exception as e:
            print("Error querying Rasa:", e)
            return {"intent": {"name": "none"}, "entities": []}

    def query(self, user_message):
        rasa_response = self.query_rasa(user_message)
        intent = rasa_response["intent"]["name"]
        entities = rasa_response["entities"]

        if intent == "query":
            return self.process_query_intent(entities)
        elif intent in self.std_responses:
            return random.choice(self.std_responses[intent])
        else:
            return "Sorry, I didn't understand that. Please try again."

    def process_query_intent(self, entities):
        row = col = None
        for ent in entities:
            if ent["entity"] == "column":
                col = ent["value"].title()
            elif ent["entity"] == "row":
                row = ent["value"].title()

        if row and col:
            try:
                value = self.df.loc[self.df[self.primarycolumnname] == row, col].values[0]
                return str(value).replace(";", ",")
            except:
                return f"Sorry, I couldn't find data for {row} and {col}."
        return "Please specify both the country and the field you're asking about."


if __name__ == "__main__":
    dfmodel = DfEngine("data/countries.csv", "Country")
    
    print(dfmodel.query("Hi"))
    print(dfmodel.query("What is Population for China?"))
    print(dfmodel.query("What is Background for Congo?"))
    print(dfmodel.query("Bye"))
