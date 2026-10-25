"""Tiny TF-IDF retriever over local markdown notes (stand-in for the paper's RAG over manuals)."""
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, folder=None):
        folder = folder or Path(__file__).resolve().parents[2] / "knowledge"
        self.docs = [l.strip("- ").strip() for p in Path(folder).glob("*.md") for l in p.read_text().splitlines()
                     if l.startswith("- ")]
        self.v = TfidfVectorizer().fit(self.docs); self.M = self.v.transform(self.docs)

    def query(self, q, k=2):
        s = cosine_similarity(self.v.transform([q]), self.M)[0]
        return [self.docs[i] for i in s.argsort()[::-1][:k]]
