# Project Concept: Vector Database for RAG Evaluation

## Overview

This project aims to establish a vector database tailored for retrieval-augmented generation (RAG) and to evaluate its performance across various configurations. The primary goal is to enhance the retrieval process for generating accurate and contextually relevant responses, particularly in scenarios involving large document sets.

## Scenario

We have created a fictitious company scenario where we have a collection of HR documents stored in the 'docs' folder. The objective is to develop an employee assistant tool that leverages these documents to provide precise answers to user queries related to HR policies and procedures. These are in `markdown` format.

## Technology Stack

The project utilizes two prominent models from the Ollama suite:

1. Embedding Model: This model is responsible for converting the HR documents into high-dimensional vectors. These vectors serve as the foundation for the vector database, enabling efficient similarity searches and retrieval of relevant documents.

2. Chat Assistant Model: Serving as the interactive component of the employee assistant, this model processes user queries and generates coherent, context-aware responses by leveraging the retrieved information from the vector database.

## Evaluation

The project includes a comprehensive evaluation framework to assess the effectiveness of different RAG configurations. By experimenting with various embedding and retrieval strategies, we aim to identify the optimal setup that maximizes the accuracy and relevance of the generated responses.

## Contribution

I encourage contributions from the community to further refine the project. Whether it's enhancing the embedding techniques, improving the retrieval algorithms, or expanding the dataset, your input is valuable in advancing this initiative.