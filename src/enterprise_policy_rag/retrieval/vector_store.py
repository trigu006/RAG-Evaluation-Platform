from langchain_chroma import Chroma
import shutil
from pathlib import Path

def create_vector_store(docs,
                        embedding_model,
                        persist_directory: str = "./data/chroma",
                        collection_name: str = "employee_handbook",
                        reset = False):
    """
    Creates a vector store from the given documents and embedding model.

    Args:
        docs (List[Document]): A list of documents to be stored in the vector store.
        embedding_model (OllamaEmbeddings): An instance of OllamaEmbeddings to generate embeddings.
        persist_directory (str): The directory where the vector store will be persisted. Defaults to "./data/chroma".
        collection_name (str): The name of the collection in the vector store. Defaults to "employee_handbook".
        reset (bool): If True, resets the existing vector store. Defaults to False.
    """

    path = Path(persist_directory)

    if reset and path.exists():
        shutil.rmtree(path)

    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )