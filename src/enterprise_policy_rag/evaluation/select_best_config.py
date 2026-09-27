"""
    Selects the best retriever configuration based on evaluation results.
    Compares the overall scores of different configurations and selects the one with the highest combined score, provided it meets a specified hit@k threshold.

    The functions should be used with the output of the `evaluate_retriever_configs` function, which evaluates different retriever configurations and returns their scores.
"""

from .retriever_config import RETRIEVER_CONFIGS

MAX_TRIVIAL_MRR_GAIN = 0.02
MAX_TRIVIAL_HIT_GAIN = 0.02
MAX_TRIVIAL_SAS_GAIN = 0.02

def select_best_config(results, k_threshold = 0.8):
    """
    Selects the best retriever configuration based on the evaluation results.

    Args:
        results (dict): A dictionary containing the evaluation results for each retriever configuration.
        k_threshold (float): The minimum hit@k score required for a configuration to be considered. Defaults to 0.8.

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
                "individual_scores": {
                    "hit_at_k": config_results["overall_scores"]["hit_at_k"],
                    "mrr": config_results["overall_scores"]["mrr"],
                    "semantic_answer_sufficiency": config_results["overall_scores"]["semantic_answer_sufficiency"]
                },
                "records": config_results["records"]
            }

    if best_config is None:
        raise ValueError("No configuration met the specified Hit@K threshold.")

    deltas = _calculate_deltas(results, best_config)

    for config_name, deltas_config in deltas.items():
        if (deltas[config_name]['absolute']['k_difference'] > 0
            and deltas[config_name]['headroom_normalized']['hit_at_k'] <= MAX_TRIVIAL_HIT_GAIN
            and deltas[config_name]['headroom_normalized']['mrr'] <= MAX_TRIVIAL_MRR_GAIN
            and deltas[config_name]['headroom_normalized']['semantic_answer_sufficiency'] <= MAX_TRIVIAL_SAS_GAIN):
            best_config = {
                "name": config_name,
                "scores": results[config_name]["overall_scores"],
                "individual_scores": {
                    "hit_at_k": results[config_name]["overall_scores"]["hit_at_k"],
                    "mrr": results[config_name]["overall_scores"]["mrr"],
                    "semantic_answer_sufficiency": results[config_name]["overall_scores"]["semantic_answer_sufficiency"]
                },
                "records": results[config_name]["records"]
            }

    return best_config, deltas

def _calculate_deltas(results, best_config):
    """
    Calculates the difference in scores between each configuration and the best configuration.

    Args:
        results (dict): A dictionary containing the evaluation results for each retriever configuration.
        best_config (dict): The best retriever configuration as returned by `select_best_config`.

    Returns:
        dict: A dictionary containing the score deltas for each configuration compared to the best configuration.
    """
    deltas = {}
    
    for config_name, config_results in results.items():
        if config_name == best_config["name"]:
            continue

        # Initialize the sub-dictionaries for absolute and normalized deltas
        deltas[config_name] = {}

        # Calculate the difference in scores for each metric
        deltas[config_name ]['absolute'] = {
            # metric: config_results["overall_scores"][metric] - best_config["scores"][metric]
            metric: best_config["scores"][metric] -config_results["overall_scores"][metric]
            for metric in best_config["scores"]
        }

        # Calculate the normalized difference in scores for each metric relative to remaining improvement potential (1 - best score)
        deltas[config_name]['headroom_normalized'] = {
            metric: deltas[config_name]['absolute'][metric] / (1 - config_results["overall_scores"][metric]) if 1 - config_results["overall_scores"][metric] != 0 else 0
            for metric in best_config["scores"]
        }

        # Calculate the difference in K thresholds if needed
        for configuration in RETRIEVER_CONFIGS:
            if configuration["name"] == config_name:
                config_params = configuration["params"]
            if configuration["name"] == best_config["name"]:
                best_config_params = configuration["params"]
        if "k" in config_params and "k" in best_config_params:
            # Absolute difference in K values
            deltas[config_name]['absolute']['k_difference'] = best_config_params["k"] - config_params["k"]
            # Percentage difference in K values
            deltas[config_name]['headroom_normalized']['k_difference'] = 0 if best_config_params["k"] == 0 else (best_config_params["k"] - config_params["k"]) / config_params["k"]

    return deltas