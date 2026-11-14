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
        return "Success - Correct abstention" if not result['predicted_answerable'] else "Failure - No abstention"

    # SECTION: Answerable questions
    # Return incorrect abstention if the generator did not answer an answerable question.
    if not result['predicted_answerable']:
        return "Failure - Incorrect abstention"

    # Retriever errors
    if result['hit_at_k_score'] == 0:
        return "Failure - Retriever did not retrieve relevant documents"

    return "Success"