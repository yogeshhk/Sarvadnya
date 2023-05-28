from text_extractor import TextExtractor
from and_other_pattern_matcher import AndOtherPatternMatcher
from such_as_pattern_matcher import SuchAsPatternMatcher
from or_other_pattern_matcher import OrOtherPatternMatcher
from including_pattern_matcher import IncludingPatternMatcher
from especially_pattern_matcher import EspeciallyPatternMatcher
from text_extractor_pipe import TextExtractorPipe
from knowledge_graph import KnowledgeGraph
from matcher_pipe import MatcherPipe
import spacy

textExtractor1 = TextExtractor("WWII", "Q362")
textExtractor1.extract()
textExtractor2 = TextExtractor("London", "Q84")
textExtractor2.extract()
textExtractor3 = TextExtractor("Paris", "Q90")
textExtractor3.extract()
textExtractor4 = TextExtractor("World War I", "Q361")
textExtractor4.extract()
textExtractorPipe = TextExtractorPipe()
textExtractorPipe.addTextExtractor(textExtractor1)
textExtractorPipe.addTextExtractor(textExtractor2)
textExtractorPipe.addTextExtractor(textExtractor3)
textExtractorPipe.addTextExtractor(textExtractor4)


nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated
doc = nlp(textExtractorPipe.extract())

andOtherPatternMatcher = AndOtherPatternMatcher(nlp)
suchAsMatcher = SuchAsPatternMatcher(nlp)
orOtherMatcher = OrOtherPatternMatcher(nlp)
includingPatternMatcher = IncludingPatternMatcher(nlp)
especiallyPatternMatcher = EspeciallyPatternMatcher(nlp)
matcherPipe = MatcherPipe()
matcherPipe.addMatcher(andOtherPatternMatcher)
matcherPipe.addMatcher(suchAsMatcher)
matcherPipe.addMatcher(orOtherMatcher)
matcherPipe.addMatcher(includingPatternMatcher)
matcherPipe.addMatcher(especiallyPatternMatcher)
relations = matcherPipe.extract(doc)

for relation in relations:
    print (relation.getHypernym(), relation.getHyponym())

knowledgeGraph = KnowledgeGraph(relations)
knowledgeGraph.build()
knowledgeGraph.show()
