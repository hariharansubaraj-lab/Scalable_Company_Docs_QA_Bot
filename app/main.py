from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid

from app.document_loader import load_pdf
from app.chunking import chunk_documents
from app.embeddings import create_embeddings
from app.vector_store import insert_documents
from app.retriever import retrieve
from app.llm import generate_answer
from app.query_rewriter import rewrite_query


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Company Docs Q&A Bot",
    description="Multi-document RAG system for PDF question answering",
    version="2.0"
)


# ---------------------------------------------------------
# Conversation History Model
# ---------------------------------------------------------

class ConversationMessage(BaseModel):

    question: str
    answer: str


# ---------------------------------------------------------
# Request Model
# ---------------------------------------------------------

class QuestionRequest(BaseModel):

    question: str

    document_id: str | None = None


# ---------------------------------------------------------
# Server-side Conversation Memory
# ---------------------------------------------------------

conversation_memory = []


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Company Docs Q&A Bot is running"
    }


# ---------------------------------------------------------
# Upload PDF
# ---------------------------------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    # ---------------------------------------------
    # Validate file
    # ---------------------------------------------

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )


    # ---------------------------------------------
    # Create unique document ID
    # ---------------------------------------------

    document_id = str(
        uuid.uuid4()
    )


    # ---------------------------------------------
    # Extract PDF text
    # ---------------------------------------------

    documents = load_pdf(
        file.file,
        file.filename
    )


    if not documents:

        raise HTTPException(
            status_code=400,
            detail="Could not extract text from the PDF."
        )


    # ---------------------------------------------
    # Create chunks
    # ---------------------------------------------

    chunks = chunk_documents(
        documents
    )


    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="Could not create document chunks."
        )


    # ---------------------------------------------
    # Create embeddings
    # ---------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts
    )


    # ---------------------------------------------
    # Store in Milvus
    # ---------------------------------------------

    inserted_count = insert_documents(
        chunks,
        embeddings,
        document_id
    )


    # ---------------------------------------------
    # Return upload information
    # ---------------------------------------------

    return {

        "message": "PDF uploaded successfully",

        "filename": file.filename,

        "document_id": document_id,

        "pages": len(documents),

        "chunks": inserted_count
    }


# ---------------------------------------------------------
# Ask Question
# ---------------------------------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    # ---------------------------------------------
    # Conversation Memory
    # ---------------------------------------------

    conversation_context = ""

    if conversation_memory:

        history_parts = []

        for message in conversation_memory:

            history_parts.append(

                f"Previous User Question: "
                f"{message['question']}\n"

                f"Previous Assistant Answer: "
                f"{message['answer']}"
            )

        conversation_context = (

            "\n\n"
            "===================="
            "\n\n"

        ).join(
            history_parts
        )


    # ---------------------------------------------
    # Query Rewriting
    # ---------------------------------------------

    print("========== CONVERSATION DEBUG ==========")

    print(
        "CURRENT QUESTION:",
        request.question
    )

    print(
        "CONVERSATION MEMORY:",
        conversation_memory
    )

    print(
        "MEMORY COUNT:",
        len(conversation_memory)
    )

    print("========================================")


    rewritten_query = rewrite_query(
        request.question,
        conversation_memory
    )


    print(
        "REWRITTEN QUERY:",
        rewritten_query
    )


    # ---------------------------------------------
    # Retrieve documents using rewritten query
    # ---------------------------------------------

    documents = retrieve(

        rewritten_query,

        document_id=request.document_id
    )


    # ---------------------------------------------
    # No relevant documents
    # ---------------------------------------------

    if not documents:

        answer = (
            "I could not find this information "
            "in the uploaded documents."
        )

        # Save conversation
        conversation_memory.append({

            "question": request.question,

            "answer": answer
        })

        return {

            "question":
                request.question,

            "rewritten_query":
                rewritten_query,

            "answer":
                answer,

            "citations": []
        }


    # ---------------------------------------------
    # Prepare document context
    # ---------------------------------------------

    context_parts = []

    citations = []

    seen_citations = set()


    for doc in documents:

        text = doc.get(
            "text",
            ""
        )

        source = doc.get(
            "source"
        )

        page = doc.get(
            "page"
        )

        score = doc.get(
            "score"
        )


        if text:

            # ---------------------------------------------
            # Add document content
            # ---------------------------------------------

            context_parts.append(

                f"Source: {source}\n"

                f"Page: {page}\n"

                f"Content:\n{text}"
            )


            # ---------------------------------------------
            # Remove duplicate citations
            # ---------------------------------------------

            citation_key = (
                source,
                page
            )


            if citation_key not in seen_citations:

                seen_citations.add(
                    citation_key
                )

                citations.append({

                    "source": source,

                    "page": page,

                    "score": score
                })


    # ---------------------------------------------
    # Combine document context
    # ---------------------------------------------

    context = (

        "\n\n"
        "===================="
        "\n\n"

    ).join(
        context_parts
    )


    # ---------------------------------------------
    # Add conversation history
    # ---------------------------------------------

    if conversation_context:

        context = (

            "Previous Conversation:\n"
            "========================================\n"

            + conversation_context

            + "\n\n"
            "Current Retrieved Document Context:\n"
            "========================================\n"

            + context
        )


    # ---------------------------------------------
    # Generate answer
    # ---------------------------------------------

    result = generate_answer(

        request.question,

        context
    )


    # ---------------------------------------------
    # Save conversation memory
    # ---------------------------------------------

    conversation_memory.append({

        "question":
            request.question,

        "answer":
            result["answer"]
    })


    # ---------------------------------------------
    # Return response
    # ---------------------------------------------

    return {

        "question":
            request.question,

        "rewritten_query":
            rewritten_query,

        "answer":
            result["answer"],

        "citations":
            citations
    }