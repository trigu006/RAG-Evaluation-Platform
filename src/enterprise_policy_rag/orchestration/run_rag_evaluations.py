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
    from src.enterprise_policy_rag.evaluation.retrieval import hit_at_k, mrr_, semantic_answer_sufficiency_at_k_

    results = []

    for question in questions:
        # Invoke retriever for sections
        sections = retriever.invoke(question['question'])

        # Invoke generator to generate answer based on retrieved sections
        response = generate_answer(sections, generator, question['question'])

        if question['answerable']:
            # Hit@K
            hit_at_k_score = hit_at_k(sections, question)  # Use the refactored hit_at_k function
            # MRR
            reciprocal_rank = mrr_(sections, question)  # Use the refactored mrr_ function
            # Semantic Answer Sufficiency at K
            best_score, hit = semantic_answer_sufficiency_at_k_(sections, question, embedding_model, THRESHOLD)  # Use the refactored function
        else:
            hit_at_k_score = None
            reciprocal_rank = None
            best_score = None
            hit = None

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
    total_hits = sum(result['hit'] for result in answerable_results)
    scores['semantic_answer_sufficiency_at_k'] = round(total_hits / len(answerable_results), 3) if len(answerable_results) > 0 else 0.0

    return scores