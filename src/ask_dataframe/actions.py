import pandas as pd
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionQueryCountryData(Action):
    def name(self) -> Text:
        return "action_query_country_data"

    def __init__(self):
        # Load once on initialization
        self.data_file = "data/countries.csv"
        self.primary_key = "Country"
        self.df = pd.read_csv(self.data_file, encoding="ISO-8859-1")

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        row = tracker.get_slot("row")
        column = tracker.get_slot("column")

        if not row or not column:
            dispatcher.utter_message(text="Please specify both a country and a column.")
            return []

        # Standardize format (title case matching your dataset)
        row = row.title()
        column = column.title()

        try:
            value = self.df.loc[self.df[self.primary_key] == row, column].item()
            value = value.replace(";", ",") if isinstance(value, str) else str(value)
            dispatcher.utter_message(text=f"The {column} for {row} is: {value}")
        except Exception as e:
            dispatcher.utter_message(text="Sorry, I couldn't find that information. Please try again.")

        return []
