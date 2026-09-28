from sentence_transformers import CrossEncoder


# ---------------------------------------------------------
# Load Reranker Model
# ---------------------------------------------------------

MODEL_NAME = "BAAI/bge-reranker-base"

reranker = CrossEncoder(
    MODEL_NAME
)


# ---------------------------------------------------------
# Re-rank Documents
# ---------------------------------------------------------

def rerank_documents(
    query: str,
    documents: list,
    top_k: int = 5
):
    """
    Re-rank retrieved documents based on
    query-document relevance.
    """

    if not documents:
        return []


    # -----------------------------------------------------
    # Create query-document pairs
    # -----------------------------------------------------

    pairs = []

    for document in documents:

        text = document.get(
            "text",
            ""
        )

        pairs.append(
            [
                query,
                text
            ]
        )


    # -----------------------------------------------------
    # Calculate reranker scores
    # -----------------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # -----------------------------------------------------
    # Add reranker score
    # -----------------------------------------------------

    for document, score in zip(
        documents,
        scores
    ):

        document["rerank_score"] = float(
            score
        )


    # -----------------------------------------------------
    # Sort by reranker score
    # -----------------------------------------------------

    ranked_documents = sorted(
        documents,
        key=lambda x: x["rerank_score"],
        reverse=True
    )


    # -----------------------------------------------------
    # Return top documents
    # -----------------------------------------------------

    return ranked_documents[:top_k]