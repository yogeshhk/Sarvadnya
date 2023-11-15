from text_extractor import TextExtractor


class TextExtractorPipe:

    __textExtractors: [TextExtractor]

    def __init__(self):
        self.__textExtractors = []

    def addTextExtractor(self, textExtractor: TextExtractor):
        self.__textExtractors.append(textExtractor)

    def extract(self) -> str:
        result = ''
        for textExtractor in self.__textExtractors:
            result = result + textExtractor.getText()
        return result