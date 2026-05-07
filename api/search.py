import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def _name_score(query: str, candidate: str, **_) -> float:
    """
    WRatio uses token_set_ratio internally, which makes nearly-identical names like
    'Very Straight 12 Inch Axle' and 'Very Bent 12 Inch Axle' score the same.
    This scorer uses ratio + token_sort_ratio so discriminating words (Straight vs Bent)
    actually matter.
    """
    q, c = query.lower().strip(), candidate.lower().strip()
    if q == c:
        return 100.0
    r = fuzz.ratio(q, c)
    ts = fuzz.token_sort_ratio(q, c)
    # partial_ratio only when query is a substring search (much shorter than candidate)
    p = fuzz.partial_ratio(q, c) if len(q) <= len(c) * 0.7 else 0
    return max(r, ts, p * 0.85)


class PartDatabase:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df.columns = ['part_name', 'location', 'usage_description', 'appearance_description']
        self.df = self.df.fillna('')

        # Include part_name in the TF-IDF corpus so description queries like
        # "12 inch axle" can match parts whose NAME contains those words.
        self.df['combined_description'] = (
            self.df['part_name'] + ' ' +
            self.df['usage_description'] + ' ' +
            self.df['appearance_description']
        )

        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_description'])

    def search_by_name(self, query: str, top_k: int = 5) -> list[dict]:
        results = process.extract(
            query,
            self.df['part_name'].tolist(),
            scorer=_name_score,
            limit=top_k * 3,
        )
        seen = set()
        output = []
        for name, score, idx in results:
            if score < 30:
                continue
            row = self.df.iloc[idx]
            key = (row['part_name'], row['location'])
            if key in seen:
                continue
            seen.add(key)
            output.append({
                'part_name': row['part_name'],
                'location': row['location'],
                'confidence': round(score / 100.0, 3),
            })
            if len(output) >= top_k:
                break
        return output

    def search_by_description(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k * 2]

        seen = set()
        output = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.05:
                break
            row = self.df.iloc[idx]
            key = (row['part_name'], row['location'])
            if key in seen:
                continue
            seen.add(key)
            output.append({
                'part_name': row['part_name'],
                'location': row['location'],
                'confidence': round(score, 3),
            })
            if len(output) >= top_k:
                break
        return output

    def get_by_name_exact(self, name: str) -> list[dict]:
        matches = self.df[self.df['part_name'] == name]
        return [
            {'part_name': row['part_name'], 'location': row['location'], 'confidence': 1.0}
            for _, row in matches.iterrows()
        ]

    def get_class_names(self) -> list[str]:
        return sorted(self.df['part_name'].unique().tolist())
