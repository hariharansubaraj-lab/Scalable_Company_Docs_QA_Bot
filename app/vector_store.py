from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

from app.config import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME
)


# =========================================================
# CONFIGURATION
# =========================================================

DIMENSION = 384


# =========================================================
# CONNECT TO MILVUS
# =========================================================

def connect_milvus():
    """
    Connect to the Milvus server.
    """

    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )


# =========================================================
# CREATE COLLECTION
# =========================================================

def create_collection():
    """
    Create the Milvus collection if it does not exist.
    """

    connect_milvus()

    fields = [

        # -------------------------------------------------
        # Primary key
        # -------------------------------------------------

        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),

        # -------------------------------------------------
        # Unique document ID
        # -------------------------------------------------

        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=100
        ),

        # -------------------------------------------------
        # Chunk text
        # -------------------------------------------------

        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=5000
        ),

        # -------------------------------------------------
        # PDF filename
        # -------------------------------------------------

        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=500
        ),

        # -------------------------------------------------
        # PDF page number
        # -------------------------------------------------

        FieldSchema(
            name="page",
            dtype=DataType.INT64
        ),

        # -------------------------------------------------
        # Embedding vector
        # -------------------------------------------------

        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=DIMENSION
        )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Uploaded company documents"
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    # -----------------------------------------------------
    # Create vector index
    # -----------------------------------------------------

    index_params = {
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "params": {}
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    collection.load()

    return collection


# =========================================================
# GET COLLECTION
# =========================================================

def get_collection():
    """
    Get the existing collection.

    If it does not exist, create it.
    """

    connect_milvus()

    if utility.has_collection(
        COLLECTION_NAME
    ):

        collection = Collection(
            COLLECTION_NAME
        )

        collection.load()

        return collection

    return create_collection()


# =========================================================
# INSERT DOCUMENT CHUNKS
# =========================================================

def insert_documents(
    chunks,
    embeddings,
    document_id
):
    """
    Insert PDF chunks and embeddings into Milvus.

    Every chunk receives the same document_id,
    allowing retrieval to filter results to
    the currently uploaded PDF.
    """

    collection = get_collection()

    # -----------------------------------------------------
    # Document IDs
    # -----------------------------------------------------

    document_ids = [
        document_id
        for _ in chunks
    ]

    # -----------------------------------------------------
    # Chunk text
    # -----------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # Source filenames
    # -----------------------------------------------------

    sources = [
        chunk["source"]
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # Page numbers
    # -----------------------------------------------------

    pages = [
        chunk["page"]
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # Prepare Milvus data
    # -----------------------------------------------------

    data = [

        document_ids,

        texts,

        sources,

        pages,

        embeddings.tolist()
    ]

    # -----------------------------------------------------
    # Insert
    # -----------------------------------------------------

    result = collection.insert(
        data
    )

    # -----------------------------------------------------
    # Flush data
    # -----------------------------------------------------

    collection.flush()

    # -----------------------------------------------------
    # Load collection
    # -----------------------------------------------------

    collection.load()

    return len(chunks)


# =========================================================
# DELETE DOCUMENT
# =========================================================

def delete_document(
    document_id: str
):
    """
    Delete all chunks belonging to a document.

    Useful when the same PDF needs to be re-uploaded
    without keeping duplicate chunks.
    """

    collection = get_collection()

    expr = (
        f'document_id == "{document_id}"'
    )

    collection.delete(
        expr
    )

    collection.flush()

    collection.load()

    return True


# =========================================================
# CHECK DOCUMENT
# =========================================================

def document_exists(
    document_id: str
):
    """
    Check whether a document exists in Milvus.
    """

    collection = get_collection()

    expr = (
        f'document_id == "{document_id}"'
    )

    results = collection.query(
        expr=expr,
        output_fields=[
            "document_id"
        ],
        limit=1
    )

    return len(results) > 0