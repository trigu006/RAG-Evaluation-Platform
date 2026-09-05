def create_retriever(vector_store, k = 3):
    """
    Create a retriever from the given vector store.

    Args:
        vector_store: The vector store to create the retriever from.
        k (int): The number of top results to return.
    """

    return vector_store.as_retriever(search_kwargs={"k": k})