

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
        for result in results:
            if not result['expected_answerable']:
                result['human_correctness'] = None
            else:
                print(f"Question ID: {result['id']}, Question: {result['question']}, Generated Answer: {result['generated_answer']}, Expected Answer: {result['expected_answer']}")
                score = input("Enter human correctness value for question [0, 1, 2]: ")
                while score not in ['0', '1', '2']:
                    print("Invalid input. Please enter 0, 1, or 2.")
                    score = input("Enter human correctness value for question [0, 1, 2]: ")
                result['human_correctness'] = int(score)
    else:
        # Since 'results' is a list of two elements (results[0] and results[1]), we need to iterate over the first element (results[0]) to add human correctness information.
        for result in results:
            if not result['expected_answerable']:
                result['human_correctness'] = None
            else:
                question_id = result['id']
                if question_id in human_correctness_dict:
                    result['human_correctness'] = human_correctness_dict[question_id]
                else:
                    print("Question ID not found in human_correctness_dict. Please provide a value.")
                    print(f"Question ID: {result['id']}, Question: {result['question']}, Generated Answer: {result['generated_answer']}, Expected Answer: {result['expected_answer']}")
                    score = input("Enter human correctness value for question [0, 1, 2]: ")
                    while score not in ['0', '1', '2']:
                        print("Invalid input. Please enter 0, 1, or 2.")
                        score = input("Enter human correctness value for question [0, 1, 2]: ")
                    result['human_correctness'] = int(score)  # or some default value if not found

    return results

def add_human_faithfulness(results, human_faithfulness_dict: dict = None):
    """
    Adds human faithfulness information to the results.

    Args:
        results (list): A list of dictionaries containing the evaluation results for each question.
        human_faithfulness_dict (dict): A dictionary mapping question IDs to human faithfulness values.

    Returns:
        list: The updated list of results with human faithfulness information added.
    """
    if human_faithfulness_dict is None:
        for result in results:
            if not result['expected_answerable']:
                result['human_faithfulness'] = None
            else:
                print(f"Question ID: {result['id']}, Question: {result['question']}, Generated Answer: {result['generated_answer']}, Sections: {result['sections']}")
                score = input("Enter human faithfulness value for question [0, 1, 2]: ")
                while score not in ['0', '1', '2']:
                    print("Invalid input. Please enter 0, 1, or 2.")
                    score = input("Enter human faithfulness value for question [0, 1, 2]: ")
                result['human_faithfulness'] = int(score)
    else:
        # Since 'results' is a list of two elements (results[0] and results[1]), we need to iterate over the first element (results[0]) to add human faithfulness information.
        for result in results:
            if not result['expected_answerable']:
                result['human_faithfulness'] = None
            else:
                question_id = result['id']
                if question_id in human_faithfulness_dict:
                    result['human_faithfulness'] = human_faithfulness_dict[question_id]
                else:
                    print("Question ID not found in human_faithfulness_dict. Please provide a value.")
                    print(f"Question ID: {result['id']}, Question: {result['question']}, Generated Answer: {result['generated_answer']}, Sections: {result['sections']}")
                    score = input("Enter human faithfulness value for question [0, 1, 2]: ")
                    while score not in ['0', '1', '2']:
                        print("Invalid input. Please enter 0, 1, or 2.")
                        score = input("Enter human faithfulness value for question [0, 1, 2]: ")
                    result['human_faithfulness'] = int(score)  # or some default value if not found

    return results