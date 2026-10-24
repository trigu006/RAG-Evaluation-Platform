

def add_human_correctness(results, human_correctness_dict: dict = None):
    """
    Adds human correctness information to the results.

    Args:
        results (list): A list of dictionaries containing the evaluation results for each question.
        human_correctness_dict (dict): A dictionary mapping question IDs to human correctness values.

    Returns:
        list: The updated list of results with human correctness information added.
    """
    if human_correctness_dict is None:
        for result in results[0]:
            print(f"Question ID: {result['id']}, Question: {result['question']}, Generated Answer: {result['generated_answer']}, Expected Answer: {result['expected_answer']}")
            result['human_correctness'] = input("Enter human correctness value for question [0, 1, 2]: ")  # or some default value if not found
    else:
        # Since 'results' is a list of two elements (results[0] and results[1]), we need to iterate over the first element (results[0]) to add human correctness information.
        for result in results[0]:
            question_id = result['id']
            if question_id in human_correctness_dict:
                result['human_correctness'] = human_correctness_dict[question_id]
            else:
                print("Question ID not found in human_correctness_dict. Please provide a value.")
                print(f"Question ID: {result['id']}, Question: {result['question']}, Generated Answer: {result['generated_answer']}, Expected Answer: {result['expected_answer']}")
                result['human_correctness'] = input("Enter human correctness value for question [0, 1, 2]: ")  # or some default value if not found
    
    return results