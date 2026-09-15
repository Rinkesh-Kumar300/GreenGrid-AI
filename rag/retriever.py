"""
GreenGrid AI -- RAG Retriever
==============================
Loads the ChromaDB vector store built by ingest.py and finds the most
relevant knowledge-base sections for any natural-language energy query.

How it works (plain English)
------------------------------
1. You provide a query such as "AC usage is high during peak hours".
2. ChromaDB converts the query into a numeric vector (embedding) using
   the same built-in model it used when storing the documents.
3. It compares that vector against every stored chunk vector using
   cosine similarity — chunks whose meaning is closest to the query
   score highest.
4. The top-N most relevant chunks are returned, each with the section
   text and the name of the source document.

This module is designed to be imported by the AI agent (Sub-Task 5)
and the FastAPI backend (Sub-Task 6):

    from rag.retriever import retrieve

    results = retrieve("AC usage is high during peak hours")
    for r in results:
        print(r["source"], "--", r["section"])
        print(r["text"])

Public API
----------
retrieve(query, n_results=3) -> list[dict]
    Returns a list of dicts sorted by relevance (most relevant first):
        source   -- filename of the source document
        title    -- document title
        section  -- section heading
        text     -- full section text
        score    -- cosine distance (lower = more relevant)

format_results(results, show_text=True) -> str
    Formats the results list as a readable string for the console or LLM.
"""

import os
import sys

import chromadb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR      = os.path.join(BASE_DIR, "rag", "chroma_db")
COLLECTION_NAME = "energy_knowledge"

# Module-level cache: the collection is loaded once and reused across calls
# so the database file is not reopened on every retrieval request.
_collection = None


def _get_collection():
    """
    Load and cache the ChromaDB collection.
    Raises a clear error if ingest.py has not been run yet.
    """
    global _collection
    if _collection is None:
        if not os.path.isdir(CHROMA_DIR):
            raise FileNotFoundError(
                f"ChromaDB store not found at {CHROMA_DIR}.\n"
                "Run  python rag/ingest.py  first to build the knowledge base."
            )
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, n_results: int = 3) -> list:
    """
    Retrieve the most relevant knowledge-base chunks for a given query.

    Parameters
    ----------
    query     : str  -- natural-language description of the energy problem
    n_results : int  -- number of results to return (default 3)

    Returns
    -------
    list of dicts sorted by relevance (most relevant first):
        source   -- source filename (e.g. "ac_efficiency.txt")
        title    -- document title
        section  -- section heading
        text     -- full section text
        score    -- cosine distance score (lower = more relevant)
    """
    collection = _get_collection()

    # ChromaDB converts the query to an embedding internally and finds
    # the closest stored chunks automatically.
    response = collection.query(
        query_texts=[query],
        n_results  =min(n_results, collection.count()),
        include    =["documents", "metadatas", "distances"],
    )

    results = []
    # response["*"][0] because we passed one query string (index 0)
    for text, meta, dist in zip(
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results.append({
            "source" : meta.get("source",  "unknown"),
            "title"  : meta.get("title",   ""),
            "section": meta.get("section", ""),
            "text"   : text,
            "score"  : round(float(dist), 4),
        })

    return results


def format_results(results: list, show_text: bool = True) -> str:
    """
    Format a list of retrieval results as a human-readable string.
    Suitable for printing to the console or passing as context to an LLM.

    Parameters
    ----------
    results   : list returned by retrieve()
    show_text : include the full section text (True) or just headings (False)
    """
    if not results:
        return "No relevant knowledge-base entries found."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[Result {i}]")
        lines.append(f"  Source  : {r['source']}")
        lines.append(f"  Section : {r['section']}")
        lines.append(f"  Score   : {r['score']}  (lower = more relevant)")
        if show_text:
            indented = "\n".join(f"  {line}" for line in r["text"].splitlines())
            lines.append(f"  Text    :\n{indented}")
        lines.append("")
    return "\n".join(lines)


# ── Command-line demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEFAULT_QUERY = (
        "Energy consumption is 18% higher than expected "
        "and AC usage is high during peak hours."
    )
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY

    print("=" * 66)
    print("  GreenGrid AI -- RAG Retriever Demo")
    print("=" * 66)
    print(f"\n  Query: \"{query}\"")
    print()

    try:
        results = retrieve(query, n_results=3)
        print(format_results(results))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
