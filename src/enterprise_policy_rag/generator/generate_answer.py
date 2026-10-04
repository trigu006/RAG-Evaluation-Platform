"""
    Generate an answer based on the retrieved information.
"""

from langchain_core.prompts import ChatPromptTemplate
from .system_prompt import SYSTEM_PROMPT

def generate_answer(retriever, generator, question):
    """
        Generate an answer based on the retrieved information.
        Inputs:
        • Retriever: the RAG retriever to the vector database
        • Generator: the language model for generating answers
        • Question: the question or message from the user
    """
    # Call the retriever to procure the sections that are relevant to the user's question
    sections = retriever.invoke(question)

    # Break 'sections' down into a single string
    sections_string = []
    #for section in sections:
    #    sections_string.append(str(section.model_dump_json()))
    #sections_string = "\n\n".join(sections_string)
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

    chain = prompt | generator

    response = chain.invoke({'context': sections_string, 'question': question})
    return response.content