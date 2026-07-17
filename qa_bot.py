#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grounded Q&A Bot with citations — повністю робоча версія.

Пайплайн (як на слайдах):
  документи → chunking (по заголовках) → embeddings → векторне сховище
  → запит → embedding → retrieval → grounding → відповідь + citations

За замовчуванням працює ПОВНІСТЮ ОФЛАЙН (лише numpy):
  • embeddings   — локальний TF-IDF (клас TfidfEmbedder)
  • vector store — InMemoryVectorStore (косинусний пошук)
  • "генерація"  — extractive: береться найрелевантніше речення з контексту

Запуск:
    python3 qa_bot.py                 # демонстрація на кількох питаннях
    python3 qa_bot.py "ваше питання"  # одне питання
"""

from __future__ import annotations
import os
import re
import sys
from pathlib import Path
import math
from dataclasses import dataclass, field
from sys import set_coroutine_origin_tracking_depth
from typing import List, Dict, Optional
from collections import Counter
import numpy as np



BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "docs")

# Поріг grounding: якщо найкращий збіг нижчий — бот каже "не знаю".
SIM_THRESHOLD = 0.22
NOT_FOUND = "Не знайшов інформації у наданій документації."


# ==================================================================
# 1) Завантаження документів
# ==================================================================
def load_documents(docs_dir: str) -> List[Dict]:
    """Читає .md / .txt. (PDF/DOCX — див. README, опційно.)"""
    docs = {}
    path = Path(docs_dir)
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in {'.txt', '.md'}:
            try:
                docs[file.name] = file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Warning! Couldn't read file {file} due to: {e}")
    return docs


# ==================================================================
# 2) Chunking — по Markdown-заголовках "## Section"
# ==================================================================
@dataclass
class Chunk:
    id: int
    source: str
    section: str
    text: str


def chunk_documents(docs: List[Dict]) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 1
    for filename, content in docs.items():
        lines = content.splitlines()
        current_section = ""
        current_text_lines = []
        for line in lines:
            if line.startswith("## "):
                previous_text = "\n".join(current_text_lines).strip()
                if previous_text:
                    chunks.append(Chunk(
                        id=chunk_id,
                        source=filename,
                        section=current_section,
                        text=previous_text
                    ))
                    chunk_id += 1
                current_section = line[3:].strip()
                current_text_lines = []
            elif line.startswith("# ") and not line.startswith("##"):
                current_section = line[2:].strip()
            else:
                current_text_lines.append(line)
        final_text = "\n".join(current_text_lines).strip()
        if final_text:
            chunks.append(Chunk(
                id=chunk_id,
                source=filename,
                section=current_section,
                text=final_text
            ))
            chunk_id += 1
    return chunks


# ==================================================================
# 3) Embeddings (за замовчуванням — локальний TF-IDF)
# ==================================================================



class TfidfEmbedder:
    """Локальний embedder без зовнішніх сервісів. Реалізує .fit() і .encode()."""

    def __init__(self):
        self.vocabulary = {}
        self.idf_vector: Optional[np.ndarray] = None
        self.num_docs= 0

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+", text.lower())

    def fit(self, texts: List[str]):
        self.num_documents = len(texts)
        if self.num_documents == 0:
            return self

        tokenized_docs = [self.tokenize(doc) for doc in texts]
        unique_words = sorted(list(set(word for doc in tokenized_docs for word in doc)))
        self.vocabulary = {word: idx for idx, word in enumerate(unique_words)}
        doc_counts = Counter()
        for doc in tokenized_docs:
            for word in set(doc):
                doc_counts[word] += 1
        idf_list = []
        for word in unique_words:
            docs_with_word = doc_counts[word]
            idf_value = math.log((1 + self.num_documents) / (1 + docs_with_word)) + 1
            idf_list.append(idf_value)
        self.idf_vector = np.array(idf_list)
        return self


    def encode(self, text: List[str]) -> np.ndarray:
        tokens = self.tokenize(text)
        vocab_size = len(self.vocabulary)
        vector = np.zeros(vocab_size)

        if len(tokens) == 0 or vocab_size == 0:
            return vector
        term_counts = Counter(tokens)
        total_tokens = len(tokens)
        tf_vector = np.zeros(vocab_size)
        for word, count in term_counts.items():
            if word in self.vocabulary:
                word_idx = self.vocabulary[word]
                tf_vector[word_idx] = count / total_tokens

        vector = tf_vector * self.idf_vector
        return vector
# ==================================================================
# 4) Векторне сховище (in-memory, косинусний пошук)
# ==================================================================
@dataclass
class Hit:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None


    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def query(self, qvec: np.ndarray, k: int = 3) -> List[Hit]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        dot_products = np.dot(self.embeddings, qvec)
        norm_embeddings = np.linalg.norm(self.embeddings, axis=1)
        norm_qvec = np.linalg.norm(qvec)

        scores = dot_products / (norm_embeddings * norm_qvec + 1e-9)
        top_indeces = np.argsort(scores)[::-1][:k]
        hits = []
        for idx in top_indeces:
            hit = Hit(
                chunk=self.chunks[idx],
                score = float(scores[idx])
            )
            hits.append(hit)

        return hits

# ==================================================================
# 5) Grounded генерація відповіді + citations
# ==================================================================
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


@dataclass
class Answer:
    text: str
    citations: List[str] = field(default_factory=list)
    grounded: bool = True
    hits: List[Hit] = field(default_factory=list)

    def __str__(self):
        if not self.citations:
            return self.text
        return f"{self.text}\n   ↳ Source: {'; '.join(self.citations)}"

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)

def generate_answer(query: str, hits: List[Hit], embedder) -> Answer:
    """За замовчуванням — extractive grounded answer.
    Бере найрелевантніше речення з top-контексту й додає citations."""
    top_hit = hits[0]
    sentences = split_sentences(top_hit.chunk.text)

    if not sentences:
        best_sentence = top_hit.chunk.text
    else:
        qvec = embedder.encode(query)
        similar = [_cosine(embedder.encode(s), qvec) for s in sentences]
        best_sentence = sentences[int(np.argmax(similar))]

    citations = [f"{top_hit.chunk.source}"]
    return Answer(text=best_sentence, citations=citations, grounded=True, hits=hits)


# ==================================================================
# 6) Сам бот
# ==================================================================
class GroundedQABot:
    def __init__(self, embedder=None, store=None, threshold: float = SIM_THRESHOLD):
        self.embedder = TfidfEmbedder()
        self.store = InMemoryVectorStore()
        self.threshold = threshold

    def index(self, docs_dir: str = DOCS_DIR):
        docs = load_documents(docs_dir)
        chunks = chunk_documents(docs)
        texts = [c.text for c in chunks]

        self.embedder.fit(texts)
        embeddings = np.array([self.embedder.encode(t) for t in texts])
        self.store.add(embeddings, chunks)
        return self

    def ask(self, question: str, k: int = 3) -> Answer:
        qvec = self.embedder.encode(question)
        hits = self.store.query(qvec, k=k)
        if not hits or hits[0].score < self.threshold:
            return Answer(text=NOT_FOUND, citations=[], grounded=False, hits=hits)
        return generate_answer(question, hits, self.embedder)

# ==================================================================
# 7) Демонстрація
# ==================================================================
DEMO_QUESTIONS = [
    "How do I reset my password?",
    "How can I create an invoice?",
    "How do I delete a user?",
    "How does API authentication work?",
    "Which currencies are supported?",
    "Do you have a mobile app?",   # немає в документації → grounding
]


def main():
    bot = GroundedQABot().index()
    print(f"Проіндексовано чанків: {len(bot.store.chunks)} "
          f"з {len(set(c.source for c in bot.store.chunks))} документів\n")

    questions = sys.argv[1:] or DEMO_QUESTIONS
    for q in questions:
        ans = bot.ask(q)
        print("Q:", q)
        print("A:", ans.text)
        if ans.citations:
            print("   ↳ Source:", "; ".join(ans.citations))
        print(f"   (score={ans.hits[0].score:.2f}, grounded={ans.grounded})" if ans.hits else "")
        print()


if __name__ == "__main__":
    main()
