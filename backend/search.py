import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class PartDatabase:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df.columns = ['part_name', 'location', 'usage_description', 'appearance_description']
        self.df = self.df.fillna('')

        self.df['combined_description'] = (
            self.df['usage_description'] + ' ' + self.df['appearance_description']
        )

        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_description'])

    def search_by_name(self, query: str, top_k: int = 5) -> list[dict]:
        results = process.extract(
            query,
            self.df['part_name'].tolist(),
            scorer=fuzz.WRatio,
            limit=top_k * 2,
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
