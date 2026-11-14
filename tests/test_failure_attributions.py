from src.enterprise_policy_rag.evaluation.classify_failures import classify_failures

def test_incorrect_abstention():
    result = {
        "expected_answerable": True,
        "predicted_answerable": False,
    }

    assert classify_failures(result) == "incorrect_abstention"

def test_correct_abstention():
    result = {
        "expected_answerable": False,
        "predicted_answerable": False,
    }

    assert classify_failures(result) == "correct_abstention"

def test_failed_abstention():
    result = {
        "expected_answerable": False,
        "predicted_answerable": True,
    }

    assert classify_failures(result) == "failed_abstention"

def test_retriever_failure():
    result = {
        "expected_answerable": True,
        "predicted_answerable": True,
        "hit_at_k_score": 0,
    }

    assert classify_failures(result) == "retriever_failure"

def test_generator_faithfulness_failure():
    result = {
        "expected_answerable": True,
        "predicted_answerable": True,
        "hit_at_k_score": 1,
        "human_faithfulness": 1,
        "human_correctness": 2,
    }

    assert classify_failures(result) == "generator_faithfulness_failure"

def test_generator_correctness_failure():
    result = {
        "expected_answerable": True,
        "predicted_answerable": True,
        "hit_at_k_score": 1,
        "human_faithfulness": 2,
        "human_correctness": 1,
    }

    assert classify_failures(result) == "generator_correctness_failure"