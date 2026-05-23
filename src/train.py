"""
src/train.py — Zero-dependency TF-IDF vectoriser and
job-role classifier.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re

from config import JOB_ROLES_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zero-dependency ML classes
# ---------------------------------------------------------------------------

class SimpleSparseMatrix:
    def __init__(self, rows, feature_names):
        self.rows = rows
        self.feature_names = feature_names
        self.shape = (len(rows), len(feature_names))

    def toarray(self):
        res = []
        for r in self.rows:
            row_arr = [r.get(w, 0.0) for w in self.feature_names]
            res.append(row_arr)
        return res


class SimpleTfidfVectorizer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = {}
        self.feature_names = []

    def fit(self, raw_documents):
        doc_counts = {}
        total_docs = len(raw_documents)
        
        tokenized_docs = []
        for doc in raw_documents:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_counts[t] = doc_counts.get(t, 0) + 1
        
        sorted_terms = sorted(list(doc_counts.keys()))
        if len(sorted_terms) > self.max_features:
            sorted_terms = sorted_terms[:self.max_features]
            
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted_terms)}
        self.feature_names = sorted_terms
        
        for word in sorted_terms:
            count = doc_counts[word]
            self.idf_[word] = math.log((1 + total_docs) / (1 + count)) + 1
            
        return self

    def transform(self, raw_documents):
        vectors = []
        for doc in raw_documents:
            tokens = self._tokenize(doc)
            tf = {}
            for t in tokens:
                if t in self.vocabulary_:
                    tf[t] = tf.get(t, 0) + 1
            
            vec = {}
            for word in self.vocabulary_:
                if word in tf:
                    vec[word] = tf[word] * self.idf_[word]
            vectors.append(vec)
        return SimpleSparseMatrix(vectors, self.feature_names)

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def _tokenize(self, text):
        return [w for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b", text.lower())]

    def get_feature_names_out(self):
        return self.feature_names


class SimpleCentroidClassifier:
    def __init__(self):
        self.classes_ = []
        self.centroids = {}

    def fit(self, X, labels):
        self.classes_ = sorted(list(set(labels)))
        self.centroids = {}
        
        for cls in self.classes_:
            class_rows = [X.rows[i] for i, label in enumerate(labels) if label == cls]
            centroid = {}
            if class_rows:
                for word in X.feature_names:
                    total_weight = sum(row.get(word, 0.0) for row in class_rows)
                    if total_weight > 0.0:
                        centroid[word] = total_weight / len(class_rows)
            self.centroids[cls] = centroid
        return self

    def predict_proba(self, X):
        results = []
        for row in X.rows:
            similarities = {}
            mag_v = math.sqrt(sum(val ** 2 for val in row.values()))
            
            for cls in self.classes_:
                centroid = self.centroids[cls]
                if mag_v == 0.0:
                    similarities[cls] = 0.0
                    continue
                
                dot = sum(row.get(word, 0.0) * centroid.get(word, 0.0) for word in row if word in centroid)
                mag_c = math.sqrt(sum(val ** 2 for val in centroid.values()))
                
                if mag_c == 0.0:
                    similarities[cls] = 0.0
                else:
                    similarities[cls] = dot / (mag_v * mag_c)
            
            total = sum(similarities.values())
            if total == 0.0:
                probs = [1.0 / len(self.classes_) for _ in self.classes_]
            else:
                probs = [similarities[cls] / total for cls in self.classes_]
            results.append(probs)
        return results


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _load_training_data() -> tuple[list[str], list[str]]:
    if not os.path.exists(JOB_ROLES_PATH):
        logger.warning(
            "job_roles.csv not found at %s — using minimal fallback data.",
            JOB_ROLES_PATH,
        )
        texts = [
            "python machine learning data analysis pandas numpy scikit-learn",
            "deep learning neural networks tensorflow pytorch gpu cuda",
            "django flask rest api postgresql redis celery backend",
            "react javascript typescript html css frontend ui ux",
            "aws gcp docker kubernetes devops ci cd terraform",
            "data visualisation tableau power bi sql reporting analyst",
        ]
        labels = [
            "Data Scientist",
            "AI Engineer",
            "Backend Developer",
            "Frontend Developer",
            "DevOps Engineer",
            "Data Analyst",
        ]
        return texts, labels

    texts = []
    labels = []
    try:
        with open(JOB_ROLES_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text") and row.get("label"):
                    texts.append(row["text"].strip())
                    labels.append(row["label"].strip())
    except Exception as exc:
        logger.error("Failed to read training data CSV: %s", exc)
        
    if not texts:
        # Final fallback so it doesn't crash
        return [
            "python ML data", "deep learning AI", "django rest API",
            "react frontend", "devops docker", "sql analytics"
        ], [
            "Data Scientist", "AI Engineer", "Backend Developer",
            "Frontend Developer", "DevOps Engineer", "Data Analyst"
        ]
        
    return texts, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model() -> tuple:
    """
    Build and fit a zero-dependency TF-IDF vectoriser + Centroid-Cosine classifier.
    """
    texts, labels = _load_training_data()
    
    vectorizer = SimpleTfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    model = SimpleCentroidClassifier()
    model.fit(X, labels)
    
    logger.info(
        "Zero-dependency model trained on %d samples with %d features.",
        len(texts),
        len(vectorizer.feature_names),
    )
    return model, vectorizer