import json
import os
import sys

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from app.retriever import retrieve
from app.query_rewriter import rewrite_query
from app.llm import generate_answer


EVALUATION_FILE = os.path.join(
    os.path.dirname(__file__),
    "evaluation_questions.json"
)


def load_questions():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def check_retrieval(
    documents,
    expected_source,
    expected_page
):

    for document in documents:

        source = document.get("source")
        page = document.get("page")

        if (
            source == expected_source
            and page == expected_page
        ):
            return True

    return False


def evaluate_question(item):

    question = item["question"]

    expected_answer = item["expected_answer"]

    expected_source = item["expected_source"]

    expected_page = item["expected_page"]

    print("\n" + "=" * 70)

    print(
        f"Question: {question}"
    )

    # --------------------------------------------------
    # Query rewriting
    # --------------------------------------------------

    rewritten_query = rewrite_query(
        question
    )

    print(
        f"Rewritten Query: {rewritten_query}"
    )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    documents = retrieve(
        rewritten_query
    )

    retrieval_correct = check_retrieval(
        documents,
        expected_source,
        expected_page
    )

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context_parts = []

    for document in documents:

        text = document.get(
            "text",
            ""
        )

        source = document.get(
            "source"
        )

        page = document.get(
            "page"
        )

        if text:

            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{text}"
            )

    context = (
        "\n\n"
        "===================="
        "\n\n"
    ).join(
        context_parts
    )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    if context:

        result = generate_answer(
            question,
            context
        )

        answer = result.get(
            "answer",
            ""
        )

    else:

        answer = (
            "No relevant context retrieved."
        )

    # --------------------------------------------------
    # Simple answer evaluation
    # --------------------------------------------------

    answer_lower = answer.lower()

    expected_lower = expected_answer.lower()

    answer_correct = (
        expected_lower in answer_lower
    )

    # --------------------------------------------------
    # Print result
    # --------------------------------------------------

    print(
        f"Expected Answer: {expected_answer}"
    )

    print(
        f"Generated Answer: {answer}"
    )

    print(
        f"Expected Source: "
        f"{expected_source}"
    )

    print(
        f"Expected Page: "
        f"{expected_page}"
    )

    print(
        f"Retrieved Chunks: "
        f"{len(documents)}"
    )

    print(
        f"Retrieval Correct: "
        f"{retrieval_correct}"
    )

    print(
        f"Answer Correct: "
        f"{answer_correct}"
    )

    return {
        "question": question,
        "retrieval_correct": retrieval_correct,
        "answer_correct": answer_correct
    }


def main():

    questions = load_questions()

    results = []

    for item in questions:

        result = evaluate_question(
            item
        )

        results.append(
            result
        )

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    total = len(results)

    retrieval_correct_count = sum(
        result["retrieval_correct"]
        for result in results
    )

    answer_correct_count = sum(
        result["answer_correct"]
        for result in results
    )

    retrieval_accuracy = (
        retrieval_correct_count
        / total
        * 100
    ) if total else 0

    answer_accuracy = (
        answer_correct_count
        / total
        * 100
    ) if total else 0

    # --------------------------------------------------
    # Final evaluation report
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RAG EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total Questions      : {total}"
    )

    print(
        f"Retrieval Correct    : "
        f"{retrieval_correct_count}"
    )

    print(
        f"Retrieval Accuracy   : "
        f"{retrieval_accuracy:.2f}%"
    )

    print(
        f"Answer Correct       : "
        f"{answer_correct_count}"
    )

    print(
        f"Answer Accuracy      : "
        f"{answer_accuracy:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()