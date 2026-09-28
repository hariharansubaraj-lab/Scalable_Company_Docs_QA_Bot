# pyrefly: ignore [missing-import]
from pymilvus import connections, utility

from app.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME


connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

if utility.has_collection(COLLECTION_NAME):

    print(f"Deleting collection: {COLLECTION_NAME}")

    utility.drop_collection(COLLECTION_NAME)

    print("Collection deleted successfully.")

else:

    print("Collection does not exist.")