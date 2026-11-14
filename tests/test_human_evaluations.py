from src.enterprise_policy_rag.evaluation.add_human_evaluations import compute_human_metrics

def test_human_faithfulness_evaluation():
    results = [
            {'id': 'sick_003',
            'question': 'How many sick days are employees entitled to per year?',
            'generated_answer': 'Six paid sick days per calendar year.',
            'expected_answer': '6 days',
            'predicted_answerable': True,
            'expected_answerable': True,
            'hit': 0,
            'hit_at_k_score': 1,
            'reciprocal_rank': 1.0,
            'human_correctness': 2,
            'human_faithfulness': 2
            },
            {'id': 'pto_004',
            'question': 'If I have five years of tenure, how many PTO days do I get per year?',
            'generated_answer': '20 days',
            'expected_answer': '20 days',
            'predicted_answerable': True,
            'expected_answerable': True,
            'hit': 0,
            'hit_at_k_score': 1,
            'reciprocal_rank': 1.0,
            'human_correctness': 2,
            'human_faithfulness': 2
            }
        ]

    human_faithfulness = compute_human_metrics(results)['human_faithfulness']
    
    assert human_faithfulness == 1

def test_human_correctness_evaluation():
    result = [
        {'id': 'sick_003',
        'question': 'How many sick days are employees entitled to per year?',
        'generated_answer': 'Six paid sick days per calendar year.',
        'expected_answer': '6 days',
        'predicted_answerable': True,
        'expected_answerable': True,
        'hit': 0,
        'hit_at_k_score': 1,
        'reciprocal_rank': 1.0,
        'human_correctness': 2,
        'human_faithfulness': 2},
        {'id': 'pto_004',
        'question': 'If I have five years of tenure, how many PTO days do I get per year?',
        'generated_answer': '20 days',
        'expected_answer': '20 days',
        'predicted_answerable': True,
        'expected_answerable': True,
        'hit': 0,
        'hit_at_k_score': 1,
        'reciprocal_rank': 1.0,
        'human_correctness': 2,
        'human_faithfulness': 2}
        ]

    human_correctness = compute_human_metrics(result)['human_correctness']

    assert human_correctness == 1