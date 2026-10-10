from pydantic import BaseModel, Field

class GeneratedAnswer(BaseModel):
    """
    Represents a generated answer with its content and associated metadata.
    """
    answer: str
    answerable: bool = Field(
        description="Indicates whether the question is answerable based on the provided context."
        )