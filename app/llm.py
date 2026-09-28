import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file"
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.5-flash-lite"


def generate_answer(question: str, context: str) -> dict:

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question ONLY from the
retrieved document context provided below.

IMPORTANT RULES:

1. Read the complete context carefully.

2. Find the information that directly answers the question.

3. If the question asks for a project title, name, date, person,
   technology, objective, or any other specific information,
   return the exact information from the context.

4. Do not replace the document's wording with your own assumption.

5. Do not use outside knowledge.

6. Do not invent information.

7. If the exact answer is present anywhere in the context,
   you MUST answer it.

8. Only say "I could not find this information in the provided
   documents." when the requested information is genuinely
   absent from the context.

9. IMPORTANT: Include an inline citation in the answer for
   every factual statement taken from the retrieved context.

10. Use this exact citation format:

   [Source: filename, Page: page_number]

11. Use the Source and Page information provided in the
    retrieved context. Do not create or modify citation details.

12. If multiple sources or pages are used, include the relevant
    citation immediately after the corresponding statement.

13. Do not add a separate "References" or "Sources" section.
    Keep citations directly inside the answer.

Retrieved Document Context:
========================================
{context}
========================================

User Question:
{question}

Answer with inline citations:
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
        store=False
    )

    answer = interaction.output_text.strip()

    return {
        "answer": answer,
        "citations": []
    }