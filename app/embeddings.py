from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


model = SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings