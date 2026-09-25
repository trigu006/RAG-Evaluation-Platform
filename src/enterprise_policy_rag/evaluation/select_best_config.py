"""
    Selects the best retriever configuration based on evaluation results.
    Compares the overall scores of different configurations and selects the one with the highest combined score, provided it meets a specified hit@k threshold.

    The functions should be used with the output of the `evaluate_retriever_configs` function, which evaluates different retriever configurations and returns their scores.
"""

def select_best_config(results, k_threshold = 0.8):
    """
    Selects the best retriever configuration based on the evaluation results.

    Args:
        results (dict): A dictionary containing the evaluation results for each retriever configuration.

    Returns:
        dict: A dictionary containing the best retriever configuration and its corresponding scores.
    """
    best_config = None
    best_score = float('-inf')

    for config_name, config_results in results.items():
        # Compare against the k_threshold to ensure hit@k is above the threshold
        if config_results["overall_scores"]["hit_at_k"] < k_threshold:
            continue

        # Calculate a combined score based on hit@k, MRR, and semantic answer sufficiency
        combined_score = (
            config_results["overall_scores"]["hit_at_k"] +
            config_results["overall_scores"]["mrr"] +
            config_results["overall_scores"]["semantic_answer_sufficiency"]
        )

        if combined_score > best_score:
            best_score = combined_score
            best_config = {
                "name": config_name,
                "scores": config_results["overall_scores"],
                "records": config_results["records"]
            }

    return best_config