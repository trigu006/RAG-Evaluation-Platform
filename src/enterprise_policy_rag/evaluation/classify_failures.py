"""
    Classifies evaluation results based on predicted answerability and human faithfulness.
"""

def classify_failures(result):
    """
    Classifies the failure type of a given evaluation result - follows a list of results, following the format of the `results` output from the evaluation pipeline (run_full_evaluations.py).

    Args:
        result (dict): A dictionary containing the evaluation result for a single question.

    Returns:
        str: A string representing the classification of the failure.
    """
    # SECTION: Unanswerable questions
    # Handle questions that are not expected to be answerable.
    if not result['expected_answerable']:
        return "correct_abstention" if not result['predicted_answerable'] else "failed_abstention"

    # SECTION: Answerable questions
    # Return incorrect abstention if the generator did not answer an answerable question.
    if not result['predicted_answerable']:
        return "incorrect_abstention"

    # Retriever errors
    if result['hit_at_k_score'] == 0:
        return "retriever_failure"

    # Generator errors
    if result['human_faithfulness'] < 2:
        return "generator_faithfulness_failure"

    if result['human_correctness'] < 2:
        return "generator_correctness_failure"

    return "success"