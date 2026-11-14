
def create_retriever(vector_store,
                     search_type: str = "similarity",
                     **search_kwargs):
    """
    Create a retriever from the given vector store.

    Args:
        vector_store: The vector store to create the retriever from.
        search_type (str): The type of search to perform.
        **search_kwargs: Additional keyword arguments for the search.
    """

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )