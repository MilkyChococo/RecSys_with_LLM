import re
import json
import numpy as np
import faiss

class QueryCleaner:
    def __init__(self, stopword_path: str):
        with open(stopword_path, "r") as f:
            self.stop_words = set(json.load(f))

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [word for word in text.split() if word not in self.stop_words]
        return " ".join(tokens)

