# Reference : Ask AI to analyse Pandas DF - PandasAI powered by Open AI
# https://www.youtube.com/watch?app=desktop&v=cGA9Yabg0Yc
# Pandas AI https://github.com/gventuri/pandas-ai

import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import random

import logging, io, json, warnings

logging.basicConfig(level="INFO")
warnings.filterwarnings('ignore')

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
# from pandasai.llm.open_assistant import OpenAssistant


class DfLLM:
    def __init__(self, data_file_name):
        self.primary_key_values_list = None
        self.data_file_name = data_file_name
        self.df = None
        self.populate_dataframe()
        self.api_key = os.environ["OPENAI_API_KEY"]
        # self.hf_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]
        self.llm = OpenAI(api_token=self.api_key)
        self.pandas_ai = PandasAI(self.llm)

        # self.oa_llm = OpenAssistant(api_token=self.hf_key)
        # self.pandas_ai_oa = PandasAI(self.oa_llm)

    def populate_dataframe(self):
        self.df = pd.read_csv(self.data_file_name, encoding="ISO-8859-1")

    def query(self, usr):
        print("User typed : " + usr)

        try:
            openai_response = self.pandas_ai.run(self.df, prompt=usr)
            print("OpenAI response : "+ openai_response)
            # assistant_response = self.pandas_ai_oa.run(self.df, prompt=usr)
            # print("Open Assistant response : " + assistant_response)
        except Exception as e:
            print(e)
            return "Could not follow your question [" + usr + "], Try again"


if __name__ == "__main__":
    data_file_name = "data/countries.csv"
    df_model = DfLLM(data_file_name)

    print(df_model.query("What is Population for China?"))
    print(df_model.query("What is Background for Congo ?"))

