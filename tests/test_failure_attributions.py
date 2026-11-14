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