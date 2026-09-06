def hit_at_k_section(questions: list, retriever) -> tuple[float, list]:
    """
    Computes the hit@k metric.

    Args:
        questions (list): A list of question dictionaries, each containing an 'id' key.
        retriever: The retriever object used to fetch predictions.

    Returns:
        tuple[float, list]: A tuple containing the hit@k score and the records.
    """

    numerator = 0
    denominator = len(questions)
    records = []

    for question in questions:
        question_id = question["id"]
        target_section = question["section_title"]
        predictions = retriever.invoke(question["question"])

        sections = [doc.metadata.get("section_title", "") for doc in predictions]
        
        if target_section in sections:
            numerator += 1

        records.append({"question_id": question_id, "score": 1 if target_section in sections else 0})

    return (numerator / denominator if denominator > 0 else 0.0, records)

def mrr(questions: list, retriever) -> tuple[float, list]:
    """
    Computes the Mean Reciprocal Rank (MRR) metric.

    Args:
        questions (list): A list of question dictionaries, each containing an 'id' key.
        retriever: The retriever object used to fetch predictions.

    Returns:
        tuple[float, list]: A tuple containing the MRR score and the records.
    """

    total_reciprocal_rank = 0
    records = []

    for question in questions:
        question_id = question["id"]
        target_section = question["section_title"]
        predictions = retriever.invoke(question["question"])

        sections = [doc.metadata.get("section_title", "") for doc in predictions]

        if target_section in sections:
            rank = sections.index(target_section) + 1  # +1 because index is 0-based
            reciprocal_rank = 1 / rank
            total_reciprocal_rank += reciprocal_rank
            records.append({"question_id": question_id, "reciprocal_rank": reciprocal_rank})
        else:
            records.append({"question_id": question_id, "reciprocal_rank": 0})

    return (total_reciprocal_rank / len(questions) if len(questions) > 0 else 0.0, records)