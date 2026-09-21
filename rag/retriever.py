"""
GreenGrid AI -- Lightweight RAG Retriever
==========================================
Retrieves the most relevant energy-saving knowledge-base sections
using TF-IDF and cosine similarity.

This version does not require ChromaDB, so it is suitable for
Vercel serverless deployment.
"""

import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "rag", "documents")

_chunks = None
_vectorizer = None
_matrix = None


def _load_chunks():
    """Load and split all knowledge-base documents into sections."""

    global _chunks

    if _chunks is not None:
        return _chunks

    chunks = []

    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(DOCS_DIR, filename)

        with open(filepath, encoding="utf-8") as f:
            raw_text = f.read()

        title_match = re.search(
            r"^TITLE:\s*(.+)$",
            raw_text,
            re.MULTILINE
        )

        title = (
            title_match.group(1).strip()
            if title_match
            else filename
        )

        parts = re.split(
            r"(^SECTION:.+$)",
            raw_text,
            flags=re.MULTILINE
        )

        i = 1
        chunk_index = 0

        while i < len(parts) - 1:

            heading = parts[i].strip()
            body = parts[i + 1].strip()

            full_text = f"{heading}\n{body}"

            if len(full_text) >= 80:

                section_name = heading.replace(
                    "SECTION:",
                    ""
                ).strip()

                chunks.append({
                    "source": filename,
                    "title": title,
                    "section": section_name,
                    "text": full_text,
                    "chunk_id": f"{filename}_{chunk_index}",
                })

                chunk_index += 1

            i += 2

    _chunks = chunks
    return _chunks


def _build_index():

    global _vectorizer
    global _matrix

    chunks = _load_chunks()

    if not chunks:
        return

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    _vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    _matrix = _vectorizer.fit_transform(documents)


def retrieve(query: str, n_results: int = 3) -> list:

    if _vectorizer is None or _matrix is None:
        _build_index()

    chunks = _load_chunks()

    if not chunks:
        return []

    query_vector = _vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        _matrix
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:n_results]:

        results.append({
            "source": chunks[index]["source"],
            "title": chunks[index]["title"],
            "section": chunks[index]["section"],
            "text": chunks[index]["text"],
            "score": round(
                float(1 - scores[index]),
                4
            ),
        })

    return results


def format_results(
    results: list,
    show_text: bool = True
) -> str:

    if not results:
        return "No relevant knowledge-base entries found."

    lines = []

    for i, result in enumerate(
        results,
        start=1
    ):

        lines.append(f"[Result {i}]")
        lines.append(
            f"  Source  : {result['source']}"
        )
        lines.append(
            f"  Section : {result['section']}"
        )
        lines.append(
            f"  Score   : {result['score']}"
        )

        if show_text:
            indented = "\n".join(
                f"  {line}"
                for line in result["text"].splitlines()
            )

            lines.append(
                f"  Text    :\n{indented}"
            )

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":

    query = (
        "Energy consumption is 18% higher "
        "than expected and AC usage is high "
        "during peak hours."
    )

    print("=" * 66)
    print("  GreenGrid AI -- RAG Retriever Demo")
    print("=" * 66)

    print(f"\nQuery: \"{query}\"\n")

    results = retrieve(
        query,
        n_results=3
    )

    print(format_results(results))