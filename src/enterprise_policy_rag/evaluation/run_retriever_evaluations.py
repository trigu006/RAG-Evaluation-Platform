from .questions import TEST_CASES
from .retrieval import hit_at_k_section, mrr, semantic_answer_sufficiency_at_k
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
        semantic_answer_sufficiency_score, semantic_answer_sufficiency_records = semantic_answer_sufficiency_at_k(TEST_CASES, retriever)

        # Store the results
        # results[retriever_name] = {
        #     "hit_at_k": {
        #         "score": hit_at_k_score,
        #         "records": hit_at_k_records
        #     },
        #     "mrr": {
        #         "score": mrr_score,
        #         "records": mrr_records
        #     },
        #     "semantic_answer_sufficiency": {
        #         "score": semantic_answer_sufficiency_score,
        #         "records": semantic_answer_sufficiency_records
        #     }
        # }
        records = []

        for question in TEST_CASES:
            question_id = question["id"]
            hit_at_k_record = next((record for record in hit_at_k_records if record["question_id"] == question_id), None)
            mrr_record = next((record for record in mrr_records if record["question_id"] == question_id), None)
            semantic_answer_sufficiency_record = next((record for record in semantic_answer_sufficiency_records if record["question_id"] == question_id), None)

            # Combine the records into a single dictionary
            combined_record = {
                "question_id": question_id,
                "hit_at_k_score": hit_at_k_record["score"] if hit_at_k_record else None,
                "mrr_score": mrr_record["reciprocal_rank"] if mrr_record else None,
                "semantic_answer_sufficiency_score": semantic_answer_sufficiency_record["best_score"] if semantic_answer_sufficiency_record else None,
                "hit": semantic_answer_sufficiency_record["hit"] if semantic_answer_sufficiency_record else None
            }

            records.append(combined_record)

        results[retriever_name] = {
            "overall_scores": {
                "hit_at_k": hit_at_k_score,
                "mrr": mrr_score,
                "semantic_answer_sufficiency": semantic_answer_sufficiency_score
            },
            "records": records
        }

    return results