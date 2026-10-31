

def run_full_evaluations(retriever, generator, questions, embedding_model, human_correctness_dict=None, human_faithfulness_dict=None):
    """
    Runs the full evaluation process, including adding human evaluations and computing metrics.

    Args:
        retriever: The retriever to evaluate.
        generator: The language model for generating answers.
        questions: A list of questions to evaluate the RAG system.
        embedding_model: The embedding model for generating embeddings.
        human_correctness_dict (dict): A dictionary mapping question IDs to human correctness values.
        human_faithfulness_dict (dict): A dictionary mapping question IDs to human faithfulness values.

    Returns:
        tuple: A tuple containing the updated results and computed human metrics.
    """
    from src.enterprise_policy_rag.evaluation.add_human_evaluations import add_human_correctness, add_human_faithfulness, compute_human_metrics
    from src.enterprise_policy_rag.orchestration.run_rag_evaluations import run_rag_evaluations

    results, scores = run_rag_evaluations(retriever, generator, questions, embedding_model)
    
    results = add_human_correctness(results, human_correctness_dict)
    results = add_human_faithfulness(results, human_faithfulness_dict)
    human_metrics = compute_human_metrics(results)

    scores.update(human_metrics)

    return results, scores