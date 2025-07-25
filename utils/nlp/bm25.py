import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Plus
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


class BM25Retriever:
    def __init__(self, labels_set: pd.DataFrame):
        self.labels_set = labels_set
        self.corpus = self._create_corpus()
        self.bm25 = BM25Plus(self.corpus)

    @staticmethod
    def _preprocess(text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [
            token
            for token in tokens
            if token not in set(stopwords.words("english"))
        ]
        return " ".join(tokens)

    def _create_corpus(self) -> List[str]:
        job_items = self.labels_set["english title"].tolist()
        similar_job_items = self.labels_set["title en"].tolist()

        job_items = [self._preprocess(x) for x in job_items]
        similar_job_items = [self._preprocess(x) for x in similar_job_items]

        corpus = [x + " " + y for x, y in zip(job_items, similar_job_items)]
        assert len(corpus) == len(job_items) == len(similar_job_items)
        return corpus

    def retrieve_relevant_items(self, query: str, top_k: int = 10) -> dict:
        scores = self.bm25.get_scores(self._preprocess(query))
        top_indices = np.argsort(scores)[::-1][:top_k]

        top_isco_codes = self.labels_set.loc[top_indices, "code"].tolist()
        return {
            "isco_codes": list(top_isco_codes),
            "indices": top_indices,
            "scores": list(scores[top_indices]),
        }
