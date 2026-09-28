import streamlit as st
import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Company Docs Q&A Bot",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "documents" not in st.session_state:

    st.session_state.documents = []


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("📄 Company Documents Q&A Bot")

st.write(
    "Upload PDFs and ask questions based on the selected document."
)


# ---------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------

st.header("1. Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Upload PDF"):

        with st.spinner(
            "Uploading and processing PDF..."
        ):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            try:

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files
                )

                if response.status_code == 200:

                    data = response.json()

                    # -------------------------------------
                    # Store document information
                    # -------------------------------------

                    document_info = {
                        "filename": data["filename"],
                        "document_id": data["document_id"]
                    }

                    st.session_state.documents.append(
                        document_info
                    )

                    st.success(
                        "PDF uploaded successfully!"
                    )

                    st.write(
                        f"**Filename:** {data['filename']}"
                    )

                    st.write(
                        f"**Pages:** {data['pages']}"
                    )

                    st.write(
                        f"**Chunks:** {data['chunks']}"
                    )

                else:

                    st.error(
                        f"Upload failed: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running."
                )


# ---------------------------------------------------------
# Uploaded Documents
# ---------------------------------------------------------

st.header("2. Uploaded Documents")


if st.session_state.documents:

    st.write(
        f"**Total uploaded documents:** "
        f"{len(st.session_state.documents)}"
    )

    for index, document in enumerate(
        st.session_state.documents,
        start=1
    ):

        st.write(
            f"{index}. 📄 **{document['filename']}**"
        )

else:

    st.info(
        "No documents uploaded in this session."
    )


# ---------------------------------------------------------
# Document Selection
# ---------------------------------------------------------

st.subheader("Select Document")

document_options = [
    "All Documents"
]

document_options.extend(
    [
        document["filename"]
        for document in st.session_state.documents
    ]
)


selected_document = st.selectbox(
    "Where should the system search?",
    document_options
)


# ---------------------------------------------------------
# Find Selected Document ID
# ---------------------------------------------------------

selected_document_id = None


if selected_document != "All Documents":

    for document in st.session_state.documents:

        if document["filename"] == selected_document:

            selected_document_id = document["document_id"]

            break


# ---------------------------------------------------------
# Display Selected Search Scope
# ---------------------------------------------------------

if selected_document == "All Documents":

    st.info(
        "🔎 Searching across all uploaded documents."
    )

else:

    st.info(
        f"🔎 Searching only in: **{selected_document}**"
    )


# ---------------------------------------------------------
# Question Section
# ---------------------------------------------------------

st.header("3. Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: Who are the project team members?"
)


if st.button("Ask Question"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching document and generating answer..."
        ):

            try:

                # -----------------------------------------
                # Prepare request
                # -----------------------------------------

                request_data = {
                    "question": question,
                    "document_id": selected_document_id
                }


                # -----------------------------------------
                # Send question to FastAPI
                # -----------------------------------------

                response = requests.post(
                    f"{API_URL}/ask",
                    json=request_data
                )


                if response.status_code == 200:

                    data = response.json()


                    # -------------------------------------
                    # Answer
                    # -------------------------------------

                    st.subheader("Answer")

                    st.write(
                        data.get(
                            "answer",
                            "No answer generated."
                        )
                    )


                    # -------------------------------------
                    # Citations
                    # -------------------------------------

                    st.subheader("Sources")

                    citations = data.get(
                        "citations",
                        []
                    )


                    if citations:

                        # ---------------------------------
                        # Remove duplicate pages
                        # ---------------------------------

                        seen_sources = set()

                        unique_citations = []


                        for citation in citations:

                            source = citation.get(
                                "source",
                                "Unknown"
                            )

                            page = citation.get(
                                "page",
                                "Unknown"
                            )

                            key = (
                                source,
                                page
                            )


                            if key in seen_sources:

                                continue


                            seen_sources.add(key)

                            unique_citations.append(
                                citation
                            )


                            # Maximum 5 unique pages
                            if len(unique_citations) >= 5:

                                break


                        # ---------------------------------
                        # Display Sources
                        # ---------------------------------

                        for citation in unique_citations:

                            source = citation.get(
                                "source",
                                "Unknown"
                            )

                            page = citation.get(
                                "page",
                                "Unknown"
                            )

                            score = citation.get(
                                "score",
                                0
                            )


                            st.write(
                                f"📄 **{source}** "
                                f"- Page {page} "
                                f"- Score: {score:.3f}"
                            )


                    else:

                        st.info(
                            "No citations available."
                        )


                else:

                    st.error(
                        f"Question failed: {response.text}"
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running."
                )