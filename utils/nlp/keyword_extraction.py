from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


class KeywordExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.sentence_model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(model=self.sentence_model)

    def extract(self, doc, keyphrase_ngram_range=(1, 1), stop_words=None):
        return self.kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
        )

# Usage
# extractor = KeywordExtractor()
# keywords = extractor.extract(doc)
# print(keywords)
