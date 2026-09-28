import re


def rewrite_query(
    question: str,
    conversation_history: list | None = None
) -> str:

    question = question.strip()

    if not question:
        return question

    if not conversation_history:
        return question

    # ---------------------------------------------------------
    # Combine ALL previous conversation messages
    # ---------------------------------------------------------

    combined_history = ""

    for message in conversation_history:

        if isinstance(message, dict):

            previous_question = message.get(
                "question",
                ""
            )

            previous_answer = message.get(
                "answer",
                ""
            )

        else:

            previous_question = getattr(
                message,
                "question",
                ""
            )

            previous_answer = getattr(
                message,
                "answer",
                ""
            )

        combined_history += (
            previous_question
            + " "
            + previous_answer
            + " "
        )

    # ---------------------------------------------------------
    # Find project title from conversation history
    # ---------------------------------------------------------

    project_match = re.search(
        r"Cloud-Enabled\s+incentive\s+management",
        combined_history,
        flags=re.IGNORECASE
    )

    project_name = None

    if project_match:
        project_name = project_match.group(0)

    # ---------------------------------------------------------
    # Resolve HIS / HER
    # ---------------------------------------------------------

    person_match = re.search(
        r"\bDr\.\s+[A-Z]\.\s*[A-Za-z]+",
        combined_history
    )

    person_name = None

    if person_match:
        person_name = person_match.group(0).strip()

    if person_name:

        rewritten_question = re.sub(
            r"\b(his|her)\b",
            person_name,
            question,
            flags=re.IGNORECASE
        )

        if rewritten_question != question:
            return rewritten_question

    # ---------------------------------------------------------
    # Resolve THIS PROJECT / THESE PROJECT
    # ---------------------------------------------------------

    if project_name:

        rewritten_question = re.sub(
            r"\b(this|these)\s+project\b",
            project_name + " project",
            question,
            flags=re.IGNORECASE
        )

        if rewritten_question != question:
            return rewritten_question

    # ---------------------------------------------------------
    # Resolve THIS / THESE
    # ---------------------------------------------------------

    if project_name:

        rewritten_question = re.sub(
            r"\b(this|these)\b",
            project_name,
            question,
            count=1,
            flags=re.IGNORECASE
        )

        if rewritten_question != question:
            return rewritten_question

    # ---------------------------------------------------------
    # Return original question if no reference found
    # ---------------------------------------------------------

    return question