"""
    Generate an answer based on the retrieved information.
"""

from langchain_core.prompts import ChatPromptTemplate
from .system_prompt import SYSTEM_PROMPT
from .answer_class import GeneratedAnswer

def generate_answer(sections, generator, question):
    """
        Generate an answer based on the retrieved information.
        Inputs:
        • Sections: the retrieved sections from the vector database
        • Generator: the language model for generating answers
        • Question: the question or message from the user
    """

    # Break 'sections' down into a single string
    sections_string = []

    sections_string = "\n\n".join(
        f"Section: {section.metadata.get('section_title', 'Unknown')}\n"
        f"{section.page_content}"
        for section in sections
        )

    # Create the prompt template for the LLM (generator)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Context: {context}\n\nQuestion: {question}")
    ])

    structured_generator = generator.with_structured_output(GeneratedAnswer, method='json_schema')

    chain = prompt | structured_generator

    response = chain.invoke({'context': sections_string, 'question': question})
    return {'answer': response.answer, 'answerable': response.answerable}