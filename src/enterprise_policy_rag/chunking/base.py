from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def split_text(text, delimiter=None, chunk_size=1000, chunk_overlap=200):
    """
    Splits the input text into chunks of specified size with overlap.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    if delimiter:
        # If a delimiter is provided, split the text using the delimiter first
        segments = text.split(delimiter)
        chunks = []
        for segment in segments:
            # Further split each segment into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                # length_function=len,
                # separators=["\n\n", "\n", " ", ""]
            )
            docs = splitter.create_documents([segment])
            chunks.extend([doc.page_content for doc in docs])
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # length_function=len,
        # separators=["\n\n", "\n", " ", ""]
    )

    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]

def markdown_chunker(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits the input text into chunks of specified size with overlap.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    headers_to_split_on = [
        ('#', 'document_title'),
        ('##', 'section_title'),
        ('###', 'subsection_title')
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # length_function=len,
        # separators=["\n\n", "\n", " ", ""]
    )

    # The markdown splitter will create documents and extract the metadata for each section based on the headers.
    # Then, the recursive character splitter will further split those sections into chunks of the specified size with overlap.
    markdown_docs = markdown_splitter.split_text(text)
    docs = splitter.split_documents(markdown_docs)
    return docs