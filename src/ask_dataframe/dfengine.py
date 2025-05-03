import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import random

import logging, io, json, warnings

logging.basicConfig(level="INFO")
warnings.filterwarnings('ignore')

import rasa_nlu
import rasa_core
import spacy

from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter


class DfEngine:
    def __init__(self, datafilename, primarycolumnname):
        self.datafilename = datafilename
        self.primarycolumnname = primarycolumnname
        self.nlu_data_filename = "data/nlu.md"
        self.nlu_model_dirname = "./models/nlu"
        self.nlu_model_name = "current"
        self.config_filename = "config/nlu_config.yml"
        self.model_directory = self.nlu_model_dirname + "/default/" + self.nlu_model_name
        self.nlu_model = None
        self.df = None
        self.d = OrderedDict()
        self.d = self.add_std_intents(self.d)
        self.build_model()

    def add_std_intents(self, d):
        d['greet'] = ["hey", "howdy", "hey there", "hello", "hi"]
        d['affirm'] = ["yes", "yep", "yeah", "indeed", "that's right", "ok", "great", "right, thank you", "correct",
                       "great choice", "sounds really good"]
        d["goodbye"] = ["bye", "goodbye", "good bye", "See you", "CU", "Chao", "Bye bye", "have a good one"]
        return d

    def populate_dataframe(self):
        self.df = pd.read_csv(self.datafilename, encoding="ISO-8859-1")
        #         self.df = self.df.dropna(axis = 1, how ='any')
        #         print(self.df.head())
        self.primary_key_values_list = self.df[self.primarycolumnname].tolist()

    def populate_nlu_training_data(self):

        if (os.path.exists(self.nlu_data_filename)):
            return

        n_sample_countries = 5
        n_sample_columns = 5
        self.all_columns = [col for col in list(self.df.columns) if col != self.primarycolumnname]
        sample_sentences = []
        for c in self.primary_key_values_list[:n_sample_countries]:
            for col in self.all_columns[:n_sample_columns]:
                if "Unnamed" in col or "NaN" in c:
                    continue
                sentenceformat1 = "What is [" + col + "](column) for [" + c + "](row) ?"
                sentenceformat2 = "For  [" + c + "](row), what is the [" + col + "](column) ?"
                sample_sentences.append(sentenceformat1)
                sample_sentences.append(sentenceformat2)

        self.d['query'] = sample_sentences

        with open(self.nlu_data_filename, 'w') as f:
            for k, v in self.d.items():
                f.write("## intent:" + k)
                for it in v:
                    f.write("\n- " + it)
                f.write("\n\n")

    def pprint(self, o):
        print(json.dumps(o, indent=2))

    def train_nlu_model(self):

        if (os.path.exists(self.model_directory)):
            return

        # loading the nlu training samples
        training_data = load_data(self.nlu_data_filename)

        # trainer to educate our pipeline
        trainer = Trainer(config.load(self.config_filename))

        # train the model!
        self.interpreter = trainer.train(training_data)

        # store it for future use
        self.model_directory = trainer.persist(self.nlu_model_dirname, fixed_model_name=self.nlu_model_name)

        self.nlu_model = Interpreter.load(self.model_directory)

    #         self.pprint(self.interpreter.parse("What is Area for Akrotiri ?"))

    def build_model(self):
        if self.nlu_model == None:
            self.populate_dataframe()
            self.populate_nlu_training_data()
            self.train_nlu_model()

    def process_query_intent(self, entities):

        row = None
        col = None
        for ent in entities:
            if ent["entity"] == "column":
                col = ent["value"].title()
            if ent["entity"] == "row":
                row = ent["value"].title()
        if row != None and col != None:
            value = self.df.loc[self.df[self.primarycolumnname] == row, col].item()
            return value.replace(";", ",")
        return "Could not follow your question. Try again"

    def process_other_intents(self, intent):
        values = self.d[intent]
        return random.choice(values)

    def query(self, usr):
        # print("User typed : " + usr)
        if self.nlu_model == None:
            self.nlu_model = Interpreter.load(self.model_directory)
            self.populate_dataframe()

        try:
            response_json = self.nlu_model.parse(usr)
            entities = response_json["entities"]
            intent = response_json['intent']['name']
            if intent == 'query':
                return self.process_query_intent(entities)
            else:
                return self.process_other_intents(intent)
        except Exception as e:
            print(e)
            return "Could not follow your question [" + usr + "], Try again"


if __name__ == "__main__":
    datafilename = "data/countries.csv"
    primarycolumnname = "Country"
    dfmodel = DfEngine(datafilename, primarycolumnname)

    response = dfmodel.query("Hi")
    print(response)

    response = dfmodel.query("What is Population for China ?")
    print(response)

    response = dfmodel.query("What is Background for Congo ?")
    print(response.encode("ISO-8859-1"))

    response = dfmodel.query("Bye")
    print(response)
