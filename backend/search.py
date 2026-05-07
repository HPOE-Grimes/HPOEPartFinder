import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def _name_score(query: str, candidate: str, **_) -> float:
    """
    WRatio uses token_set_ratio internally, which makes nearly-identical names like
    'Very Straight 12 Inch Axle' and 'Very Bent 12 Inch Axle' score the same.
    This scorer uses ratio + token_sort_ratio so discriminating words actually matter.
    """
    q, c = query.lower().strip(), candidate.lower().strip()
    if q == c:
        return 100.0
    r = fuzz.ratio(q, c)
    ts = fuzz.token_sort_ratio(q, c)
    p = fuzz.partial_ratio(q, c) if len(q) <= len(c) * 0.7 else 0
    return max(r, ts, p * 0.85)


class PartDatabase:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df.columns = ['part_name', 'location', 'usage_description', 'appearance_description']
        self.df = self.df.fillna('')

        self.df['combined_description'] = (
            self.df['part_name'] + '. ' +
            self.df['usage_description'] + ' ' +
            self.df['appearance_description']
        )

        print("Loading semantic embedding model...")
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
        self._embeddings = self._model.encode(
            self.df['combined_description'].tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        print("Embedding model ready.")

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
        query_vec = self._model.encode([query], convert_to_numpy=True)
        semantic_scores = cosine_similarity(query_vec, self._embeddings).flatten()

        # Keyword boost: score against usage+appearance only (not part name) so a
        # part named "Metal Shaft Insert" doesn't get free points just from its name
        # containing query words like "metal" and "shaft".
        stop = {'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'it', 'is',
                'and', 'or', 'for', 'with', 'that', 'had', 'go', 'be', 'was'}
        query_words = {w for w in query.lower().split() if w not in stop and len(w) > 2}
        desc_only = (self.df['usage_description'] + ' ' + self.df['appearance_description'])
        keyword_scores = np.array([
            sum(1 for w in query_words if w in desc.lower()) / max(len(query_words), 1)
            for desc in desc_only
        ])

        combined = semantic_scores * 0.55 + keyword_scores * 0.45
        top_indices = combined.argsort()[::-1][:top_k * 2]

        seen = set()
        output = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            key = (row['part_name'], row['location'])
            if key in seen:
                continue
            seen.add(key)
            output.append({
                'part_name': row['part_name'],
                'location': row['location'],
                'confidence': round(float(combined[idx]), 3),
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
