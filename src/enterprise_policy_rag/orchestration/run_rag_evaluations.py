from src.enterprise_policy_rag.generator.generate_answer import generate_answer

THRESHOLD = 0.8

def run_rag_evaluations(retriever, generator, questions, embedding_model):
    """
    Run evaluations for the RAG system based on a set of questions.

    Args:
        retriever: The retriever to evaluate.
        generator: The language model for generating answers.
        questions: A list of questions to evaluate the RAG system.
        embedding_model: The embedding model for generating embeddings.
    Returns:
        dict: A dictionary containing the evaluation results for the RAG system.
    """
    from langchain_chroma.vectorstores import cosine_similarity
    from src.enterprise_policy_rag.evaluation.retrieval import hit_at_k, mrr_

    results = []

    for question in questions:
        # Invoke retriever for sections
        sections = retriever.invoke(question['question'])

        # Invoke generator to generate answer based on retrieved sections
        response = generate_answer(sections, generator, question['question'])

        # Assess retriever performance
        expected_vector = embedding_model.embed_query(question['answer'])
        chunk_vectors = embedding_model.embed_documents([section.page_content for section in sections])

        # Compute cosine similarity between expected vector and chunk vectors
        cos_sim = cosine_similarity([expected_vector], chunk_vectors)[0]
        best_score = max(cos_sim, default=0.0)
        hit = int(best_score >= THRESHOLD)  # Using the provided threshold for sufficiency

        if question['answerable']:
            # Hit@K
            hit_at_k_score = hit_at_k(sections, question)  # Use the refactored hit_at_k function
            # MRR
            reciprocal_rank = mrr_(sections, question)  # Use the refactored mrr_ function
        else:
            hit_at_k_score = None
            reciprocal_rank = None

        # Append the evaluation results for the current question
        results.append({'id': question['id'],
                        'question': question['question'],
                        'answer': response['answer'],
                        'predicted_answerable': response['answerable'],
                        'expected_answerable': question['answerable'],
                        'best_score': best_score,
                        'hit': hit,
                        'hit_at_k_score': hit_at_k_score,
                        'reciprocal_rank': reciprocal_rank,
                        'sections': sections})

    scores = _compute_rag_metrics(results)

    return results, scores

def _compute_rag_metrics(results):
    """
    Compute evaluation metrics for the RAG system based on the results.

    Args:
        results: A list of evaluation results for each question.
    Returns:
        dict: A dictionary containing the computed evaluation metrics for the RAG system.
    """
    scores = {}
    answerable_results = [result for result in results if result['hit_at_k_score'] is not None and result['reciprocal_rank'] is not None]

    # Compute answerability accuracy
    answerability_score = sum(1 for result in results if result['predicted_answerable'] == result['expected_answerable'])
    scores['answerability_accuracy'] = round(answerability_score / len(results), 3)

    # Compute Hit@K accuracy
    hit_at_k_score = sum(result['hit_at_k_score'] for result in results if result['hit_at_k_score'] is not None)
    scores['hit_at_k_accuracy'] = round(hit_at_k_score / len(answerable_results), 3)

    # Compute Mean Reciprocal Rank (MRR)
    total_reciprocal_rank = sum(result['reciprocal_rank'] for result in results if result['reciprocal_rank'] is not None)
    scores['mean_reciprocal_rank'] = round(total_reciprocal_rank / len(answerable_results), 3)

    # Compute semantic answer sufficiency at K
    total_hits = sum(result['hit'] for result in results)
    scores['semantic_answer_sufficiency_at_k'] = round(total_hits / len(results), 3)

    return scores