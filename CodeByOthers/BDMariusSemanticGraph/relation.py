class Relation:

    __hypernym: str
    __hyponym: str

    def __init__(self, hypernym, hyponym):
        self.__hypernym = hypernym
        self.__hyponym = hyponym

    def getHypernym(self):
        return self.__hypernym

    def getHyponym(self):
        return self.__hyponym

