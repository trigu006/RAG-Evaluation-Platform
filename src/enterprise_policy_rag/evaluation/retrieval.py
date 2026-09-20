from langchain_chroma.vectorstores import cosine_similarity

from src.enterprise_policy_rag.embeddings.ollama_embeddings import get_embedding_model

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

def semantic_answer_sufficiency_at_k(questions: list, retriever, embedding_model = None, threshold = 0.75) -> tuple[float, list]:
    """
    Computes the semantic answer sufficiency at k metric.

    Args:
        questions (list): A list of question dictionaries, each containing an 'id' key.
        retriever: The retriever object used to fetch predictions.
        embedding_model: The embedding model used to encode the answers and predictions.
        threshold (float): The similarity threshold for determining answer sufficiency.

    Returns:
        tuple[float, list]: A tuple containing the semantic answer sufficiency score and the records.
    """

    # Instantiate the embedding model to encode the answers
    if embedding_model is None:
        embedding_model = get_embedding_model(
            model_name="embeddinggemma"
        )

    records = []
    total_hits = 0

    for question in questions:
        question_id = question["id"]
        answer = question["answer"]

        # Get the list of docs from the question
        docs = retriever.invoke(question["question"])

        # Encode the answer to get its vector representation
        expected_vector = embedding_model.embed_query(answer)

        chunk_vectors = embedding_model.embed_documents(
            [doc.page_content for doc in docs]
        )

        # Calculate cosine similarity between the expected answer vector and each of the chunk vectors
        similarities = cosine_similarity(
            [expected_vector],
            chunk_vectors
        )[0]

        best_score = max(similarities, default=0.0)
        hit = int(best_score >= threshold)  # Using the provided threshold for sufficiency

        total_hits += hit

        records.append({
                "question_id": question_id,
                "best_score": best_score,
                "hit": hit
            })

    return (total_hits / len(questions) if len(questions) > 0 else 0.0, records)