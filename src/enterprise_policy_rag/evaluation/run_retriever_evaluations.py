from .questions import TEST_CASES
from .retrieval import hit_at_k_section, mrr
from .retriever_config import RETRIEVER_CONFIGS
from src.enterprise_policy_rag.retrieval.retriever import create_retriever

def evaluate_retriever_configs(vector_store):
    """
    Evaluates different retriever configurations using the provided vector store.

    Args:
        vector_store: The vector store to create the retriever from.

    Returns:
        dict: A dictionary containing the evaluation results for each retriever configuration.
    """
    results = {}

    for config in RETRIEVER_CONFIGS:
        retriever_name = config["name"]
        retriever_params = config["params"]

        # Create the retriever with the specified parameters
        retriever = create_retriever(vector_store, **retriever_params)

        # Evaluate hit@k and MRR
        hit_at_k_score, hit_at_k_records = hit_at_k_section(TEST_CASES, retriever)
        mrr_score, mrr_records = mrr(TEST_CASES, retriever)

        # Store the results
        results[retriever_name] = {
            "hit_at_k": {
                "score": hit_at_k_score,
                "records": hit_at_k_records
            },
            "mrr": {
                "score": mrr_score,
                "records": mrr_records
            }
        }

    return results