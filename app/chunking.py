from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents):

    chunks = []

    for document in documents:

        text = document["text"].strip()

        if not text:
            continue

        # First split using paragraphs
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        current_chunk = ""

        for paragraph in paragraphs:

            # If adding the paragraph stays within chunk size
            if len(current_chunk) + len(paragraph) + 2 <= CHUNK_SIZE:

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

            else:

                if current_chunk:

                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": document["source"],
                        "page": document["page"]
                    })

                # Handle paragraphs larger than CHUNK_SIZE
                if len(paragraph) > CHUNK_SIZE:

                    start = 0

                    while start < len(paragraph):

                        end = start + CHUNK_SIZE

                        small_chunk = paragraph[start:end].strip()

                        if small_chunk:

                            chunks.append({
                                "text": small_chunk,
                                "source": document["source"],
                                "page": document["page"]
                            })

                        start = end - CHUNK_OVERLAP

                    current_chunk = ""

                else:
                    current_chunk = paragraph

        # Add remaining text
        if current_chunk:

            chunks.append({
                "text": current_chunk.strip(),
                "source": document["source"],
                "page": document["page"]
            })

    return chunks