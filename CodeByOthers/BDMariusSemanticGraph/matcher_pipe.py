from pattern_matcher import PatternMatcher
from relation import Relation
from spacy.tokens import Doc


class MatcherPipe:

    __matchers: [PatternMatcher]

    def __init__(self):
        self.__matchers = []

    def addMatcher(self, matcher: PatternMatcher):
        self.__matchers.append(matcher)

    def extract(self, doc: Doc) -> [Relation]:
        results = []
        for matcher in self.__matchers:
            results.extend(matcher.getRelations(doc))
        return results
