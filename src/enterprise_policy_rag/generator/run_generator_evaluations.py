"""
    Run evaluations for the generator based on a set of questions.
"""

from .generate_answer import generate_answer

def run_generator_evaluations(retriever, generator, questions):
    """
        Run evaluations for the generator based on a set of questions.
        Inputs:
        • Retriever: the RAG retriever to the vector database
        • Generator: the language model for generating answers
        • Questions: a list of questions to evaluate the generator
    """
    results = []

    for question in questions:
        # Refactored to pass the sections
        sections = retriever.invoke(question['question'])
        
        # answer = generate_answer(retriever, generator, question['question'])
        answer = generate_answer(sections, generator, question['question'])
        results.append({'id': question['id'],'question': question['question'], 'answer': answer, 'sections': sections})
        
    return results