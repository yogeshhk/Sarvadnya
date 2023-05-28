"""
  A wrapper class to interface with neo4j graph database.
  The interface is acting as "memory".
"""

from neo4jrestclient.client import GraphDatabase
from neo4jrestclient import client
from random import randrange


class BotMemory:
    """
        Simple wrapper class.
        allows to add to memory.
        perform simple inference.
    """
    def __init__(self, db_user, db_pass, host="http://localhost:7474"):
        # connect to the database.
        self.db = GraphDatabase(host,
                                username=db_user,
                                password=db_pass)

    def memorize_restaurant_facts(self, name, cuisine=None, location=None):
        """
           A method to memorize facts related to restaurants
        """
        n = self._add_node(name, "restaurant_name")
        if cuisine is not None:
            c = self._add_node(cuisine, "cuisine")
            n.relationships.create("is_a", c)
        if location is not None:
            l = self._add_node(location, "location")
            n.relationships.create("located", l)

    def memorize_user_facts(self, name, cuisine=None, location=None):
        """
           A method to memorize facts about user.
        """
        n = self._add_node(name, "user_name")
        if cuisine is not None:
            c = self._add_node(cuisine, "cuisine")
            n.relationships.create("likes", c)
        if location is not None:
            l = self._add_node(location, "location")
            n.relationships.create("located", l)

    def clear_memory(self):
        """
           forget everything.
        """
        q = "MATCH (n:user_name) DETACH DELETE n"
        results = self.db.query(q, returns=(client.Node, str, client.Node))
        q = "MATCH (n:restaurant_name) DETACH DELETE n"
        results = self.db.query(q, returns=(client.Node, str, client.Node))
        q = "MATCH (n:cuisine) DETACH DELETE n"
        results = self.db.query(q, returns=(client.Node, str, client.Node))
        q = "MATCH (n:location) DETACH DELETE n"
        results = self.db.query(q, returns=(client.Node, str, client.Node))

    def _add_node(self, node_name, node_type):
        """
            add a node if not existed.
            and return the node.
        """
        q = 'MATCH (r:' + node_type + ') WHERE r.name="' \
            + node_name + '" RETURN r'
        results = self.db.query(q, returns=(client.Node, str, client.Node))
        res = self.db.labels.create(node_type)

        if (len(results) == 0):
            r = self.db.nodes.create(name=node_name)
            res.add(r)
        else:
            r = results[0][0]
        return r

    def get_user_location(self, u):
        loc = None
        q = 'MATCH p=(u:user_name {name:"' + u + '"})-[:located]->(ul:location) RETURN ul'
        results = self.db.query(q, data_contents=True)
        if len(results) > 0:
            loc = results.rows[0][0]["name"]
        return loc

    def find_restaurant(self, u, c):
        """
           find a good choice!
           Should handle when there is no result or
           when user location is not specified.
           If cuisine c specified apply it.
        """
        r_name = None
        r_location = None
        r_cuisine = None
        u_name = u

        if c is not None:
            q0 = 'MATCH p=(rl:location)<-[:located]-(res:restaurant_name)-[r:is_a]->(c:cuisine {name:"' + c + '"})<-[:likes]-(u:user_name {name:"' + u + '"})-[:located]->(ul:location) where ul=rl RETURN res, c, u, rl  LIMIT 25'
            results = self.db.query(q0, data_contents=True)
            if (len(results) > 0):
                random_index = randrange(0, len(results))
                r_name = results.rows[random_index][0]["name"]
                r_location = results.rows[random_index][3]["name"]
                r_cuisine = results.rows[random_index][1]["name"]
                return (u_name, r_name, r_location, r_cuisine)

        q1 = 'MATCH p=(rl:location)<-[:located]-(res:restaurant_name)-[r:is_a]->(c:cuisine)<-[:likes]-(u:user_name {name:"' + u + '"})-[:located]->(ul:location) where ul=rl RETURN res, c, u, rl  LIMIT 25'

        results = self.db.query(q1, data_contents=True)
        if (len(results) > 0):
            random_index = randrange(0, len(results))
            r_name = results.rows[random_index][0]["name"]
            r_location = results.rows[random_index][3]["name"]
            r_cuisine = results.rows[random_index][1]["name"]
            return (u_name, r_name, r_location, r_cuisine)

        q2 = 'MATCH p=(u:user_name {name:"' + u + '"})-[:located]->(ul:location)<-[:located]-(res:restaurant_name)  RETURN res, u, ul  LIMIT 25'
        results = self.db.query(q2, data_contents=True)
        if (len(results) > 0):
            random_index = randrange(0, len(results))
            r_name = results.rows[random_index][0]["name"]
            r_location = results.rows[random_index][2]["name"]

        return (u_name, r_name, r_location, r_cuisine)

