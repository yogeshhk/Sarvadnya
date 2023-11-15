from pattern_matcher import PatternMatcher
from spacy.tokens import Doc
from relation import Relation


class EspeciallyPatternMatcher(PatternMatcher):

    def __init__(self, nlp):
        pattern = [{'POS': 'NOUN'},
                   {'IS_PUNCT': True, 'OP': '?'},
                   {'LOWER': 'especially'},
                   {'POS': 'NOUN'}]
        PatternMatcher.__init__(self, pattern, nlp, "especially")

    def getRelations(self, doc: Doc) -> [Relation]:
        relations = []
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            candidates = set()
            for sent in doc.sents:
                for token in sent:
                    # Find relation
                    if token.i == span.root.i:
                        for token2 in sent:
                            # First hyponym
                            if token2.head.i == token.i:
                                for token3 in sent:
                                    startToken = token3
                                    while startToken and startToken.head.i != sent.root.i and startToken.i != token2.i:
                                        if startToken.pos_ == "NOUN":
                                            candidates.add(startToken)
                                        startToken = startToken.head
            if len(candidates) > 0:
                hypernym = span.text.split()[0].replace(',', '')
                for candidate in candidates:
                    relations.append(Relation(hypernym, candidate.text))

        return relations


