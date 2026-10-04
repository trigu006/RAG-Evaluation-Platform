"""
    Generate an answer based on the retrieved information.
"""

from langchain_core.prompts import ChatPromptTemplate

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
    # NEED TO FIX: Currently, the iteration of sections is composed of 'Document' types, resulting in an error.
    sections_string = "\n".join(sections)

    # Create the prompt template for the LLM (generator)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "Context: {context}\n\nQuestion: {question}")
    ])

    chain = prompt | generator

    response = chain.invoke({'context': sections_string, 'question': question})
    return response