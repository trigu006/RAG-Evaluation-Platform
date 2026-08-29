from langchain_chroma import Chroma

def create_vector_store(docs, embedding_model, persist_directory: str = "./data/chroma", collection_name: str = "employee_handbook"):
    """
    Creates a vector store from the given documents and embedding model.

    Args:
        docs (List[Document]): A list of documents to be stored in the vector store.
        embedding_model (OllamaEmbeddings): An instance of OllamaEmbeddings to generate embeddings.
    """
    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )