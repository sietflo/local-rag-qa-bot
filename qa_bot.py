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
from typing import List, Dict, Optional
from collections import Counter
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "docs")

# Поріг grounding: якщо найкращий збіг нижчий — бот каже "не знаю".
SIM_THRESHOLD = 0.22
NOT_FOUND = "Не знайшов інформації у наданій документації."


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
load_dotenv()
client = genai.Client(api_key=GEMINI_API_KEY)

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
# 3) Embeddings
# ==================================================================


class GeminiEmbedder:
    """Генерація векторів через офіційне API Google."""

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type
                )
            )
            # Повертаємо вектор як numpy array
            return np.array(result.embeddings[0].values)
        except Exception as e:
            print(f"API Error generating embedding: {e}")
            # fallback на нульовий вектор у разі помилки мережі
            return np.zeros(768)
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

    def query(self, qvec: np.ndarray, k: int = 3, threshold: float = 0.0) -> List[Hit]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        dot_products = np.dot(self.embeddings, qvec)
        norm_embeddings = np.linalg.norm(self.embeddings, axis=1)
        norm_qvec = np.linalg.norm(qvec)

        scores = dot_products / (norm_embeddings * norm_qvec + 1e-9)
        valid_indices = np.where(scores >= threshold)[0]
        if len(valid_indices) == 0:
            return []
        sorted_valid = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        top_indeces = sorted_valid[:k]

        hits = []
        for idx in top_indeces:
            hit = Hit(
                chunk=self.chunks[idx],
                score=float(scores[idx])
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


def generate_answer(query: str, hits: List[Hit]) -> Answer:
    """Генерує фінальну відповідь на основі знайденого контексту."""
    if not hits:
        return Answer(text=NOT_FOUND, citations=None, grounded=False, hits=hits)
    context_parts = [
        f"[Source: {hit.chunk.source}, Section: {hit.chunk.section}]\n{hit.chunk.text}"
        for hit in hits
    ]
    context = "\n\n---\n\n".join(context_parts)
    system_instruction = (
        "You are a helpful grounded QA assistant. "
        "Answer the user's question strictly using only the provided context. "
        "If the context does not contain the answer, reply exactly with: "
        "\"Не знайшов інформації у наданій документації.\". "
        "Do not make up facts or use external knowledge."
        "Instead of the raw fact from the document, improve it to be less robotic to user"
        "Every response should be strictly in English"
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2
        )
    )
    top_hit = hits[0]
    answer_text = response.text.strip()
    if answer_text.startswith(NOT_FOUND):
        return Answer(text=answer_text, citations=None, grounded=False, hits=hits)
    citations = [f"{top_hit.chunk.source}"]
    return Answer(text=answer_text, citations=citations, grounded=True, hits=hits)

# ==================================================================
# 6) Сам бот
# ==================================================================
class GroundedQABot:
    def __init__(self, embedder=None, store=None, threshold: float = SIM_THRESHOLD):
        self.embedder = GeminiEmbedder()
        self.store = InMemoryVectorStore()
        self.threshold = threshold

    def index(self, docs_dir: str = DOCS_DIR):
        docs = load_documents(docs_dir)
        chunks = chunk_documents(docs)

        embeddings_list = []
        for c in chunks:
            vec = self.embedder.embed_text(c.text, task_type="RETRIEVAL_DOCUMENT")
            embeddings_list.append(vec)
        embeddings = np.array(embeddings_list)
        self.store.add(embeddings, chunks)
        return self

    def ask(self, question: str, k: int = 3) -> Answer:
        qvec = self.embedder.embed_text(question, task_type="RETRIEVAL_QUERY")
        hits = self.store.query(qvec, k=k, threshold= self.threshold)
        return generate_answer(question, hits)

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
