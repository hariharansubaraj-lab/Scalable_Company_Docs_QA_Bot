import re
import json

from app.embeddings import create_embeddings
from app.vector_store import get_collection
from app.config import TOP_K
from app.reranker import rerank_documents
from app.bm25_retriever import bm25_search


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:

    text = str(text or "").lower()

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

    return text.strip()


# =========================================================
# STOP WORDS
# =========================================================

STOP_WORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "tell",
    "me",
    "about",
    "please",
    "give",
    "show",
    "can",
    "you",
    "do",
    "does",
    "did",
    "this",
    "that",
    "these",
    "those",
}


# =========================================================
# FACTUAL QUERY WORDS
# =========================================================

FACTUAL_TERMS = {
    "team",
    "teams",
    "member",
    "members",
    "guide",
    "mentor",
    "leader",
    "name",
    "names",
    "developer",
    "developers",
    "student",
    "students",
    "person",
    "persons",
    "people",
    "author",
    "authors",
}


# =========================================================
# QUERY TYPE DETECTION
# =========================================================

def is_team_member_query(query: str) -> bool:

    normalized = normalize_text(query)

    return (
        "team member" in normalized
        or "team members" in normalized
        or "project member" in normalized
        or "project members" in normalized
        or normalized.strip() == "team"
        or normalized.strip() == "members"
    )


def is_guide_query(query: str) -> bool:

    normalized = normalize_text(query)

    return (
        "project guide" in normalized
        or "guide name" in normalized
        or "who is the guide" in normalized
        or normalized.strip() == "guide"
    )


# =========================================================
# MULTI SEARCH QUERY GENERATION
# =========================================================

def get_search_queries(query: str):

    normalized = normalize_text(query)

    # -----------------------------------------------------
    # TEAM MEMBER QUERY
    # -----------------------------------------------------

    if is_team_member_query(query):

        return [
            query,
            "project team members",
            "student names",
            "students who carried out the project",
            "project report authors",
            "submitted by",
            "project students",
            "bonafide certificate"
        ]

    # -----------------------------------------------------
    # GUIDE QUERY
    # -----------------------------------------------------

    if is_guide_query(query):

        return [
            query,
            "project guide name",
            "guided by",
            "project supervision",
            "project guide",
            "bonafide certificate"
        ]

    # -----------------------------------------------------
    # STUDENT / AUTHOR QUERY
    # -----------------------------------------------------

    if (
        "student name" in normalized
        or "student names" in normalized
        or "author name" in normalized
        or "author names" in normalized
        or "who submitted" in normalized
    ):

        return [
            query,
            "student names",
            "project authors",
            "students who carried out the project",
            "submitted by",
            "bonafide certificate"
        ]

    return [query]


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text: str):

    words = normalize_text(text).split()

    return [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 2
    ]


# =========================================================
# PAGE NUMBER EXTRACTION
# =========================================================

def extract_page_number(query: str):

    query_lower = query.lower().strip()

    patterns = [
        r"\bpage\s+(\d+)\b",
        r"\bpage\s+no\.?\s+(\d+)\b",
        r"\bpage\s+number\.?\s+(\d+)\b",
        r"\bpageno\.?\s+(\d+)\b",
        r"\bpage\s+#\s*(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query_lower
        )

        if match:

            return int(
                match.group(1)
            )

    return None


# =========================================================
# KEYWORD SCORE
# =========================================================

def keyword_score(
    query: str,
    document_text: str
) -> float:

    query_keywords = extract_keywords(
        query
    )

    if not query_keywords:

        return 0.0

    document_words = set(
        normalize_text(
            document_text
        ).split()
    )

    matched = 0

    for keyword in query_keywords:

        if keyword in document_words:

            matched += 1

    return matched / len(
        query_keywords
    )


# =========================================================
# PHRASE SCORE
# =========================================================

def phrase_score(
    query: str,
    document_text: str
) -> float:

    keywords = extract_keywords(
        query
    )

    if len(keywords) < 2:

        return 0.0

    normalized_document = normalize_text(
        document_text
    )

    important_phrase = " ".join(
        keywords
    )

    if important_phrase in normalized_document:

        return 1.0

    matched = 0

    for keyword in keywords:

        if keyword in normalized_document:

            matched += 1

    if matched >= 2:

        return 0.5

    return 0.0


# =========================================================
# FACTUAL QUERY DETECTION
# =========================================================

def is_factual_query(query: str) -> bool:

    query_words = set(
        extract_keywords(query)
    )

    return bool(
        query_words.intersection(
            FACTUAL_TERMS
        )
    )


# =========================================================
# FACTUAL BOOST
# =========================================================

def factual_boost(
    query: str,
    text: str
) -> float:

    if not is_factual_query(query):

        return 0.0

    query_words = set(
        extract_keywords(query)
    )

    text_words = set(
        normalize_text(text).split()
    )

    matched = query_words.intersection(
        text_words
    )

    if not matched:

        return 0.0

    return min(
        len(matched) * 0.10,
        0.40
    )


# =========================================================
# DUPLICATE KEY
# =========================================================

def create_duplicate_key(document):

    return (
        document.get("document_id"),
        document.get("source"),
        document.get("page"),
        normalize_text(
            document.get(
                "text",
                ""
            )
        )
    )


# =========================================================
# PAGE RETRIEVAL
# =========================================================

def retrieve_by_page(
    page_number: int,
    document_id: str | None = None,
    source: str | None = None
):

    collection = get_collection()

    filters = [
        f"page == {page_number}"
    ]

    if document_id:

        filters.append(
            f"document_id == "
            f"{json.dumps(document_id)}"
        )

    if source:

        filters.append(
            f"source == "
            f"{json.dumps(source)}"
        )

    expr = " and ".join(
        filters
    )

    results = collection.query(
        expr=expr,
        output_fields=[
            "document_id",
            "text",
            "source",
            "page"
        ]
    )

    documents = []

    seen = set()

    for result in results:

        text = result.get(
            "text",
            ""
        )

        if not text:

            continue

        document = {

            "document_id":
                result.get(
                    "document_id"
                ),

            "text":
                text,

            "source":
                result.get(
                    "source"
                ),

            "page":
                result.get(
                    "page"
                ),

            "score": 1.0,

            "semantic_score": 1.0,

            "lexical_score": 1.0,

            "bm25_score": 1.0,

            "rerank_score": 1.0
        }

        key = create_duplicate_key(
            document
        )

        if key in seen:

            continue

        seen.add(key)

        documents.append(
            document
        )

    return documents


# =========================================================
# MAIN RETRIEVER
# =========================================================

def retrieve(
    query: str,
    document_id: str | None = None,
    source: str | None = None
):

    collection = get_collection()

    # =====================================================
    # NORMALIZE DOCUMENT ID
    # =====================================================

    if document_id in (
        None,
        "",
        "null",
        "None"
    ):

        document_id = None

    # =====================================================
    # PAGE-SPECIFIC QUERY
    # =====================================================

    page_number = extract_page_number(
        query
    )

    if page_number is not None:

        return retrieve_by_page(
            page_number=page_number,
            document_id=document_id,
            source=source
        )

    # =====================================================
    # GET MULTIPLE SEARCH QUERIES
    # =====================================================

    search_queries = get_search_queries(
        query
    )

    print(
        f"Original query: {query}"
    )

    print(
        f"Search queries: {search_queries}"
    )

    # =====================================================
    # MILVUS FILTER
    # =====================================================

    filters = []

    if document_id:

        filters.append(
            f"document_id == "
            f"{json.dumps(document_id)}"
        )

    if source:

        filters.append(
            f"source == "
            f"{json.dumps(source)}"
        )

    expr = None

    if filters:

        expr = " and ".join(
            filters
        )

    # =====================================================
    # CANDIDATES
    # =====================================================

    candidates = []

    # =====================================================
    # VECTOR / SEMANTIC SEARCH
    # =====================================================

    semantic_limit = max(
        TOP_K * 5,
        50
    )

    for search_query in search_queries:

        try:

            query_embedding = create_embeddings(
                [search_query]
            )[0]

            search_results = collection.search(

                data=[
                    query_embedding.tolist()
                ],

                anns_field="embedding",

                param={
                    "metric_type": "COSINE",
                    "params": {}
                },

                limit=semantic_limit,

                expr=expr,

                output_fields=[
                    "document_id",
                    "text",
                    "source",
                    "page"
                ]
            )

            for hits in search_results:

                for hit in hits:

                    entity = hit.entity

                    text = entity.get(
                        "text",
                        ""
                    )

                    if not text:

                        continue

                    document = {

                        "document_id":
                            entity.get(
                                "document_id"
                            ),

                        "text":
                            text,

                        "source":
                            entity.get(
                                "source"
                            ),

                        "page":
                            entity.get(
                                "page"
                            )
                    }

                    key = create_duplicate_key(
                        document
                    )

                    semantic_score = float(
                        hit.distance
                    )

                    lexical = keyword_score(
                        search_query,
                        text
                    )

                    phrase = phrase_score(
                        search_query,
                        text
                    )

                    boost = factual_boost(
                        query,
                        text
                    )

                    hybrid_score = (

                        semantic_score * 0.55

                        +

                        lexical * 0.25

                        +

                        phrase * 0.10

                        +

                        boost * 0.10
                    )

                    document.update({

                        "score":
                            hybrid_score,

                        "semantic_score":
                            semantic_score,

                        "lexical_score":
                            lexical,

                        "phrase_score":
                            phrase,

                        "factual_boost":
                            boost,

                        "bm25_score":
                            0.0
                    })

                    existing = None

                    for candidate in candidates:

                        if (
                            create_duplicate_key(
                                candidate
                            )
                            == key
                        ):

                            existing = candidate

                            break

                    if existing is None:

                        candidates.append(
                            document
                        )

                    else:

                        if (
                            document["score"]
                            >
                            existing["score"]
                        ):

                            existing.update(
                                document
                            )

        except Exception as error:

            print(
                f"Semantic search error for "
                f"'{search_query}':",
                error
            )

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    try:

        bm25_results = bm25_search(
            query=query,
            document_id=document_id,
            top_k=max(
                TOP_K * 5,
                20
            )
        )

        print(
            f"BM25 results: {len(bm25_results)}"
        )

        # -------------------------------------------------
        # Normalize BM25 scores
        # -------------------------------------------------

        max_bm25_score = 0.0

        if bm25_results:

            max_bm25_score = max(
                float(
                    result.get(
                        "bm25_score",
                        0.0
                    )
                )
                for result in bm25_results
            )

        # -------------------------------------------------
        # Add BM25 results
        # -------------------------------------------------

        for result in bm25_results:

            text = result.get(
                "text",
                ""
            )

            if not text:

                continue

            document = {

                "document_id":
                    result.get(
                        "document_id"
                    ),

                "text":
                    text,

                "source":
                    result.get(
                        "source"
                    ),

                "page":
                    result.get(
                        "page"
                    )
            }

            key = create_duplicate_key(
                document
            )

            raw_bm25_score = float(
                result.get(
                    "bm25_score",
                    0.0
                )
            )

            if max_bm25_score > 0:

                normalized_bm25 = (
                    raw_bm25_score
                    /
                    max_bm25_score
                )

            else:

                normalized_bm25 = 0.0

            lexical = keyword_score(
                query,
                text
            )

            phrase = phrase_score(
                query,
                text
            )

            boost = factual_boost(
                query,
                text
            )

            # -------------------------------------------------
            # Hybrid score
            #
            # Vector      = 50%
            # BM25        = 30%
            # Keyword     = 10%
            # Phrase      = 5%
            # Factual     = 5%
            # -------------------------------------------------

            existing = None

            for candidate in candidates:

                if (
                    create_duplicate_key(
                        candidate
                    )
                    == key
                ):

                    existing = candidate

                    break

            if existing is not None:

                existing["bm25_score"] = (
                    normalized_bm25
                )

                old_semantic = float(
                    existing.get(
                        "semantic_score",
                        0.0
                    )
                )

                combined_score = (

                    old_semantic * 0.50

                    +

                    normalized_bm25 * 0.30

                    +

                    lexical * 0.10

                    +

                    phrase * 0.05

                    +

                    boost * 0.05
                )

                existing["score"] = (
                    combined_score
                )

            else:

                document.update({

                    "score":
                        (
                            normalized_bm25 * 0.30
                            +
                            lexical * 0.10
                            +
                            phrase * 0.05
                            +
                            boost * 0.05
                        ),

                    "semantic_score":
                        0.0,

                    "lexical_score":
                        lexical,

                    "phrase_score":
                        phrase,

                    "factual_boost":
                        boost,

                    "bm25_score":
                        normalized_bm25
                })

                candidates.append(
                    document
                )

    except Exception as error:

        print(
            "BM25 search error:",
            error
        )

    # =====================================================
    # KEYWORD FALLBACK
    # =====================================================

    fallback_query = query

    if is_team_member_query(query):

        fallback_query = (
            "project team members "
            "student names "
            "students who carried out the project "
            "submitted by "
            "project authors"
        )

    elif is_guide_query(query):

        fallback_query = (
            "project guide "
            "guide name "
            "guided by "
            "project supervision"
        )

    query_keywords = extract_keywords(
        fallback_query
    )

    if query_keywords:

        try:

            fallback_filters = []

            if document_id:

                fallback_filters.append(
                    f"document_id == "
                    f"{json.dumps(document_id)}"
                )

            if source:

                fallback_filters.append(
                    f"source == "
                    f"{json.dumps(source)}"
                )

            if fallback_filters:

                fallback_expr = (
                    " and ".join(
                        fallback_filters
                    )
                )

            else:

                fallback_expr = (
                    'document_id != ""'
                )

            fallback_results = collection.query(

                expr=fallback_expr,

                output_fields=[
                    "document_id",
                    "text",
                    "source",
                    "page"
                ],

                limit=1000
            )

            for entity in fallback_results:

                text = entity.get(
                    "text",
                    ""
                )

                if not text:

                    continue

                document = {

                    "document_id":
                        entity.get(
                            "document_id"
                        ),

                    "text":
                        text,

                    "source":
                        entity.get(
                            "source"
                        ),

                    "page":
                        entity.get(
                            "page"
                        )
                }

                lexical = keyword_score(
                    fallback_query,
                    text
                )

                phrase = phrase_score(
                    fallback_query,
                    text
                )

                boost = factual_boost(
                    query,
                    text
                )

                if (
                    lexical < 0.20
                    and phrase < 0.50
                    and boost <= 0.0
                ):

                    continue

                key = create_duplicate_key(
                    document
                )

                already_exists = False

                for candidate in candidates:

                    if (
                        create_duplicate_key(
                            candidate
                        )
                        == key
                    ):

                        already_exists = True

                        if lexical > candidate.get(
                            "lexical_score",
                            0.0
                        ):

                            candidate.update({

                                "score":
                                    max(
                                        candidate.get(
                                            "score",
                                            0.0
                                        ),
                                        lexical * 0.55
                                        +
                                        phrase * 0.25
                                        +
                                        boost * 0.20
                                    ),

                                "lexical_score":
                                    lexical,

                                "phrase_score":
                                    phrase,

                                "factual_boost":
                                    boost
                            })

                        break

                if already_exists:

                    continue

                document.update({

                    "score":
                        lexical * 0.55
                        +
                        phrase * 0.25
                        +
                        boost * 0.20,

                    "semantic_score":
                        0.0,

                    "lexical_score":
                        lexical,

                    "phrase_score":
                        phrase,

                    "factual_boost":
                        boost,

                    "bm25_score":
                        0.0
                })

                candidates.append(
                    document
                )

        except Exception as error:

            print(
                "Keyword fallback error:",
                error
            )

    # =====================================================
    # NO RESULTS
    # =====================================================

    if not candidates:

        return []

    # =====================================================
    # SORT BEFORE RERANKING
    # =====================================================

    candidates.sort(

        key=lambda x:
            x.get(
                "score",
                0.0
            ),

        reverse=True
    )

    # =====================================================
    # KEEP MORE CANDIDATES
    # =====================================================

    rerank_input = candidates[
        :max(
            TOP_K * 4,
            15
        )
    ]

    # =====================================================
    # RERANKING
    # =====================================================

    try:

        ranked_candidates = rerank_documents(

            query=query,

            documents=rerank_input,

            top_k=TOP_K
        )

    except Exception as error:

        print(
            "Reranker error:",
            error
        )

        ranked_candidates = (
            rerank_input[:TOP_K]
        )

    # =====================================================
    # PROTECT STRONG KEYWORD MATCHES
    # =====================================================

    strong_matches = []

    for candidate in candidates:

        lexical = float(
            candidate.get(
                "lexical_score",
                0.0
            )
        )

        phrase = float(
            candidate.get(
                "phrase_score",
                0.0
            )
        )

        boost = float(
            candidate.get(
                "factual_boost",
                0.0
            )
        )

        if (
            lexical >= 0.60
            or phrase >= 0.50
            or boost >= 0.20
        ):

            strong_matches.append(
                candidate
            )

    # =====================================================
    # ADD STRONG MATCHES BACK
    # =====================================================

    ranked_keys = {

        create_duplicate_key(
            document
        )

        for document in ranked_candidates
    }

    for candidate in strong_matches:

        key = create_duplicate_key(
            candidate
        )

        if key not in ranked_keys:

            ranked_candidates.append(
                candidate
            )

            ranked_keys.add(
                key
            )

    # =====================================================
    # FINAL SORT
    # =====================================================

    def final_score(document):

        rerank = float(
            document.get(
                "rerank_score",
                0.0
            )
        )

        lexical = float(
            document.get(
                "lexical_score",
                0.0
            )
        )

        phrase = float(
            document.get(
                "phrase_score",
                0.0
            )
        )

        boost = float(
            document.get(
                "factual_boost",
                0.0
            )
        )

        bm25 = float(
            document.get(
                "bm25_score",
                0.0
            )
        )

        return max(

            rerank,

            lexical * 0.80,

            phrase * 0.90,

            boost,

            bm25 * 0.75
        )

    ranked_candidates.sort(

        key=final_score,

        reverse=True
    )

    # =====================================================
    # FINAL TOP-K
    # =====================================================

    ranked_candidates = (
        ranked_candidates[:TOP_K]
    )

    # =====================================================
    # PREPARE RESPONSE
    # =====================================================

    documents = []

    for candidate in ranked_candidates:

        rerank_score = float(

            candidate.get(

                "rerank_score",

                candidate.get(
                    "score",
                    0.0
                )
            )
        )

        documents.append({

            "document_id":
                candidate.get(
                    "document_id"
                ),

            "text":
                candidate.get(
                    "text",
                    ""
                ),

            "source":
                candidate.get(
                    "source"
                ),

            "page":
                candidate.get(
                    "page"
                ),

            "score":
                rerank_score,

            "semantic_score":
                float(
                    candidate.get(
                        "semantic_score",
                        0.0
                    )
                ),

            "lexical_score":
                float(
                    candidate.get(
                        "lexical_score",
                        0.0
                    )
                ),

            "bm25_score":
                float(
                    candidate.get(
                        "bm25_score",
                        0.0
                    )
                ),

            "rerank_score":
                rerank_score
        })

    return documents