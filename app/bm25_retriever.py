from rank_bm25 import BM25Okapi
import re

from app.vector_store import get_collection


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

def tokenize(text: str):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().split()


# ---------------------------------------------------------
# Get documents from Milvus
# ---------------------------------------------------------

def get_all_chunks(
    document_id: str | None = None
):

    collection = get_collection()

    # Make sure collection is loaded
    collection.load()

    if document_id:

        expr = (
            f'document_id == "{document_id}"'
        )

    else:

        expr = "id >= 0"

    results = collection.query(
        expr=expr,
        output_fields=[
            "document_id",
            "text",
            "source",
            "page"
        ],
        limit=16384
    )

    return results


# ---------------------------------------------------------
# BM25 Search
# ---------------------------------------------------------

def bm25_search(
    query: str,
    document_id: str | None = None,
    top_k: int = 10
):

    chunks = get_all_chunks(
        document_id
    )

    if not chunks:
        return []

    tokenized_corpus = [
        tokenize(
            chunk.get("text", "")
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indexes[:top_k]:

        score = float(
            scores[index]
        )

        if score <= 0:
            continue

        result = dict(
            chunks[index]
        )

        result["bm25_score"] = score

        results.append(
            result
        )

    return results