from pattern_matcher import PatternMatcher
from spacy.tokens import Doc
from relation import Relation


class IncludingPatternMatcher(PatternMatcher):

    def __init__(self, nlp):
        pattern = [{'POS': 'NOUN'},
                   {'IS_PUNCT': True, 'OP': '?'},
                   {'LOWER': 'including'},
                   {'POS': 'NOUN'}]
        PatternMatcher.__init__(self, pattern, nlp, "including")

    def getRelations(self, doc: Doc) -> [Relation]:
        relations = []
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            for sent in doc.sents:
                for token in sent:
                    # Find the relation
                    if token.text == "including" and token.head.i == span.root.i:
                        for token2 in sent:
                            # First hyponym
                            if token2.head.i == token.i:
                                results = set()
                                results.add(span.text.split()[-1])
                                # Other hyponyms
                                for token3 in sent:
                                    startToken = token3
                                    while startToken and startToken.head.i != sent.root.i and startToken.i != token2.i:
                                        if startToken.pos_ == "NOUN":
                                            results.add(startToken.text)
                                        startToken = startToken.head
                                if len(results) > 0:
                                    hypernym = span.text.split()[0].replace(',', '')
                                    for result in results:
                                        relations.append(Relation(hypernym, result))
        return relations


