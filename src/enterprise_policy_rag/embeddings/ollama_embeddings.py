from langchain_ollama import OllamaEmbeddings

def get_embedding_model(model_name: str = "embeddinggemma") -> OllamaEmbeddings:
    """
    Returns an instance of OllamaEmbeddings.

    Returns:
        OllamaEmbeddings: An instance of OllamaEmbeddings.
    """
    return OllamaEmbeddings(
        model=model_name,
    )