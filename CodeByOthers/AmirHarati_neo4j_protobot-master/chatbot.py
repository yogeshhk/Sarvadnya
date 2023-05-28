"""
  Basic commandline chatbot to test NLP/neo4j pipeline.
"""
from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.config import RasaNLUConfig
from neo4jrestclient.client import GraphDatabase
import warnings
import BotMemory as bm
import BotResponse as br

db_user = "protobot"
db_pass = "12345"

bot_mem = bm.BotMemory(db_user, db_pass)


# where `model_directory points to the folder the model is persisted in
interpreter = Interpreter.load("./projects/default/model_20180402-153148/", RasaNLUConfig("config_spacy.json"))


# this is a very limited and simple bot.
# It starts always with Hi
output_str = "Ready!"
last_user = "amir" # this  is also the default user
last_location = "san francisco"
while(1 == 1):
    print("> ", output_str)
    input_str = input()
    # turnoff the warning related to numpy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        results = interpreter.parse(input_str)

    rname = results["intent"]["name"]
    rconf = results["intent"]["confidence"]
    # print(results)
    # if input is a fact about user
    if rname == "i_fact" and rconf >= 0.5:
        # for now it is always one user.
        # TODO: learn the user name through chat
        u = None
        c = None
        l = None
        for e in results["entities"]:
            if e["entity"] == "user_name":
                u = e["value"]
            if e["entity"] == "cuisine":
                c = e["value"]
            if e["entity"] == "location":
                l = e["value"]
        if u is not None:
            last_user = u
        else:
            # default user since we always need a user
            u = last_user
        if l is not None:
            last_location = l
        else:
            pass
            # default location since we always need a location
            # BUG: we dont handle when user have more than 1 locaton

        output_str = br.user_fact_res(u, c, l)
        output_str += "\nI will make sure I remember next time I see you!"
        bot_mem.memorize_user_facts(u, c, l)

        pass
    # if input is a fact about a restaurant
    elif rname == "r_fact" and rconf >= 0.5:
        n = None
        c = None
        l = None
        for e in results["entities"]:
            if e["entity"] == "restaurant_name":
                n = e["value"]
            if e["entity"] == "cuisine":
                c = e["value"]
            if e["entity"] == "location":
                l = e["value"]
        output_str = br.restaurant_fact_res(n, c, l)
        output_str += "\nI will make sure I remember next time I see you!"
        bot_mem.memorize_restaurant_facts(n, c, l)
    # if input is a query for a resturant
    elif rname == "restaurant_search" and rconf >= 0.5:
        n = None
        c = None
        l = None
        for e in results["entities"]:
            if e["entity"] == "restaurant_name":
                n = e["value"]
            if e["entity"] == "cuisine":
                c = e["value"]
            if e["entity"] == "location":
                l = e["value"]
        # user should specify their own location.
        if bot_mem.get_user_location(last_user) is None:
            output_str = "OH! I don't know which city you live. Can you tell me?"
        else:
            un, rn, loc, cus = bot_mem.find_restaurant(last_user, c)
            output_str = br.find_restaurant_res(rn, loc, cus,
                                                last_location)

    elif (input_str.strip() == "clear"):
        output_str = "OK! I forget everything you said to me!"
        bot_mem.clear_memory()

    elif rname == "affirm" and rconf >= 0.3:
        output_str = "Sure!"

    elif rname == "greet" and rconf > 0.5:
        output_str = "Hi"

    elif rname == "goodbye" and rconf > 0.5:
        print("Bye")
        break
    else:
        output_str = 'I am not sure what you mean'
