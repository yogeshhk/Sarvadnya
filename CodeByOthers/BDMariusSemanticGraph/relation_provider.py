from relation import Relation

class RelationProvider:

    __relations: [Relation]

    def __init__(self, relations=[Relation]):
        self.__relations = relations

    def getRelations(self):
        return self.__relations
