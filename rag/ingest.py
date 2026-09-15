"""
GreenGrid AI -- RAG Ingest Pipeline
=====================================
Reads the plain-text knowledge-base documents in rag/documents/,
splits them into sections, and stores every section as a searchable
chunk inside a local ChromaDB vector database.

This script only needs to be run ONCE (or re-run when documents change).
The database is saved to rag/chroma_db/ and persists between runs.

How it works (plain English)
------------------------------
1. Read each .txt file in rag/documents/.
2. Split each file into sections using "SECTION:" headings as dividers.
   Each section becomes one independent, retrievable chunk.
3. Give every chunk a unique ID and store which file it came from
   (the "source" metadata field used later to show where text came from).
4. Pass all chunks to ChromaDB, which automatically converts each chunk
   into a numeric vector (embedding) and stores both the text and the
   vector together.
5. When the retriever later searches for relevant chunks, ChromaDB
   compares the search query's vector against all stored vectors and
   returns the closest matches by cosine similarity.

Usage:
    python rag/ingest.py
"""

import os
import sys
import re

import chromadb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR        = os.path.join(BASE_DIR, "rag", "documents")
CHROMA_DIR      = os.path.join(BASE_DIR, "rag", "chroma_db")
COLLECTION_NAME = "energy_knowledge"

# Minimum characters a chunk must contain to be worth storing.
MIN_CHUNK_LENGTH = 80


def load_documents(docs_dir: str) -> list:
    """
    Read every .txt file in docs_dir.

    Returns a list of dicts, each with:
        filename  -- e.g. "ac_efficiency.txt"
        title     -- extracted from the TITLE: header line, or filename as fallback
        raw_text  -- full file content as a string
    """
    docs = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            raw_text = f.read()

        # Extract the TITLE: line from the top of the file
        title_match = re.search(r"^TITLE:\s*(.+)$", raw_text, re.MULTILINE)
        title       = title_match.group(1).strip() if title_match else filename

        docs.append({"filename": filename, "title": title, "raw_text": raw_text})
        print(f"  Loaded: {filename}  ({len(raw_text):,} chars)")
    return docs


def split_into_chunks(doc: dict) -> list:
    """
    Split one document into topic-sized chunks by splitting on SECTION: headings.

    Each returned chunk dict has:
        chunk_id  -- unique ID such as "ac_efficiency_3"
        source    -- filename (shown to users as the source of a retrieved tip)
        title     -- document title
        section   -- heading of this section
        text      -- full text of this section (heading + body paragraph)
    """
    raw   = doc["raw_text"]

    # re.split with a capturing group keeps the SECTION: line in the results list.
    # Result layout: [preamble, "SECTION: Heading", body, "SECTION: Heading", body, ...]
    parts = re.split(r"(^SECTION:.+$)", raw, flags=re.MULTILINE)

    chunks      = []
    chunk_index = 0

    # Start from index 1 to skip the preamble before the first SECTION:
    i = 1
    while i < len(parts) - 1:
        heading   = parts[i].strip()          # e.g. "SECTION: Turn Off Unused Equipment"
        body      = parts[i + 1].strip()      # the paragraph that follows
        full_text = f"{heading}\n{body}"

        if len(full_text) >= MIN_CHUNK_LENGTH:
            section_name = heading.replace("SECTION:", "").strip()
            chunks.append({
                "chunk_id": f"{doc['filename'].replace('.txt', '')}_{chunk_index}",
                "source"  : doc["filename"],
                "title"   : doc["title"],
                "section" : section_name,
                "text"    : full_text,
            })
            chunk_index += 1

        i += 2  # jump to the next SECTION: heading

    return chunks


def build_vectorstore(docs: list) -> int:
    """
    Store all chunks in a ChromaDB persistent collection on disk.

    ChromaDB handles embeddings automatically using its built-in model —
    no separate embeddings service is required.

    Returns the total number of chunks stored.
    """
    # PersistentClient saves the database to CHROMA_DIR on disk so it
    # survives between Python sessions.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete and recreate the collection so re-running ingest is always clean.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"\n  Deleted existing collection '{COLLECTION_NAME}' (rebuilding)")

    # cosine similarity is better than Euclidean distance for text search
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    for doc in docs:
        chunks = split_into_chunks(doc)
        all_chunks.extend(chunks)
        print(f"  Chunked: {doc['filename']}  ->  {len(chunks)} sections")

    if not all_chunks:
        print("ERROR: no chunks produced. Check the documents directory.")
        return 0

    # Add all chunks to ChromaDB in one batch — ChromaDB auto-generates embeddings
    collection.add(
        ids       = [c["chunk_id"] for c in all_chunks],
        documents = [c["text"]     for c in all_chunks],
        metadatas = [
            {"source": c["source"], "title": c["title"], "section": c["section"]}
            for c in all_chunks
        ],
    )
    return len(all_chunks)


def main():
    print("=" * 62)
    print("  GreenGrid AI -- RAG Ingest")
    print("=" * 62)

    if not os.path.isdir(DOCS_DIR):
        print(f"ERROR: documents directory not found at {DOCS_DIR}")
        sys.exit(1)

    print(f"\nLoading documents from: {DOCS_DIR}")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("ERROR: no .txt files found.")
        sys.exit(1)

    print(f"\nBuilding vector store ...")
    total = build_vectorstore(docs)

    print(f"\nDone. Stored {total} chunks in: {CHROMA_DIR}")
    print("Run  python rag/retriever.py  to test semantic search.")


if __name__ == "__main__":
    main()
