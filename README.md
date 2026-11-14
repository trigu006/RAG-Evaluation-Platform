# Enterprise Policy RAG Evaluation Platform

A RAG evaluation project designed to benchmark retrieval strategies for enterprise HR policies. The repository creates a realistic employee-assistant scenario using policy documents in Markdown format, stores them in a Chroma vector database, and measures how well different retrieval configurations answer policy-related questions.

More importantly, the project evaluates the performance of the RAG pipeline across various metrics to understand where failures occur.

## Overview

This project demonstrates a practical retrieval-augmented generation workflow for enterprise knowledge systems. It combines:

- local embeddings from Ollama
- a Chroma-backed vector store
- multiple retriever configurations
- evaluation metrics for retrieval and answer quality
- a generator step that uses retrieved sections to answer questions

The goal is not just to build a chatbot - we need to evaluate which RAG setup is most effective for policy lookup and question answering.

## Scenario

The repository models a fictional company, Northstar Analytics, that stores employee policy information in the `docs/` folder. These documents cover topics such as:

- 401(k) benefits
- parental leave
- paid time off
- employee handbook policies
- workplace procedures and benefit summaries

The objective is to provide an employee-facing assistant that can answer HR questions using policy text as the source of truth.

## Project Structure

```text
.
├── docs/                          # HR policy documents in Markdown
├── data/                         # Persisted vector store artifacts
├── src/
│   └── enterprise_policy_rag/
│       ├── chunking/             # Chunking utilities
│       ├── embeddings/           # Ollama embedding integration
│       ├── evaluation/           # Question sets and retrieval metrics
│       ├── generator/            # Answer generation logic
│       ├── orchestration/        # End-to-end evaluation workflows
│       ├── retrieval/            # Vector store and retriever creation
│       └── __init__.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── tests
│   ├── test_failure_attributions.py    # Test failure classifications
│   ├── test_human_evaluations.py       # Test human evaluations
│   ├── test_orchestration.py           # Test orchestration
│   └── test_retrieval_metrics.py       # Test retrieval metrics
├── test.ipynb                   # exploratory notebook / validation work
└── .gitignore
```

## Technology Stack

This project uses a local, open-source stack built around LangChain and Ollama:

- Python 3.12+
- LangChain / LangChain Chroma integration
- Chroma for vector storage and retrieval
- Ollama embeddings (`embeddinggemma` by default)
- Ollama chat model for answer generation
- retrieval evaluation metrics such as Hit@K, MRR, and semantic answer sufficiency

## Data and Evaluation Focus

The project includes a curated set of evaluation questions in `src/enterprise_policy_rag/evaluation/questions.py`. These questions cover both answerable and non-answerable queries and are designed to test whether the retriever finds the correct policy sections and whether the generated answer is faithful to the source documents.

The evaluation workflow measures:

- Hit@K: whether the correct section appears among the top retrieved results
- MRR: how early the correct section appears in ranked retrieval results
- Semantic answer sufficiency: whether retrieved passages are semantically close to the expected answer
- Answerability accuracy: whether the system correctly recognizes when a question cannot be answered from policy text
- Faithfulness: whether the generator adheres to the information presented in the sections from the retriever
- Correctness: whether the generator's response is correct
- Failure attribution: which component is associated with an incorrect response

## Key Components

### Embeddings
The embedding layer creates vector representations of policy sections so they can be compared semantically across queries.

### Vector Store
The vector store persists document chunks and supports similarity and MMR-based retrieval strategies.

### Retrievers
Multiple retriever configurations are evaluated, including different values of `k` and search types.

### Generator
The generator uses the retrieved policy sections as context and produces a response from the selected Ollama chat model.

### Evaluation Pipeline
The orchestration layer runs retrieval experiments and aggregates the final metrics used to compare system quality across different configurations.

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you hit import-path issues when running Python from the repo root, set:

```bash
export PYTHONPATH=.
```

### 2. Install and run Ollama

Make sure Ollama is installed and running locally.

```bash
ollama serve
```

Pull the models you plan to use. The default embedding model is `embeddinggemma`, and you can use any locally available chat model supported by Ollama for answer generation.

```bash
ollama pull embeddinggemma
ollama pull llama3.1
```

> You can swap in other supported Ollama models depending on your environment and evaluation goals.

### 3. Prepare the document corpus

Place your policy documents in `docs/` as Markdown files. The project expects policy content that can be chunked and embedded for retrieval.

### 4. Run the evaluation pipeline

The project is organized around Python modules rather than a single CLI script. Use the functions in the `src/enterprise_policy_rag` package from a notebook or Python script to:

- generate embeddings
- create the vector store
- build retrievers
- evaluate retrieval metrics
- run full RAG evaluations

## Example Workflow

The repository includes reusable building blocks such as:

- `get_embedding_model()`
- `create_vector_store()`
- `create_retriever()`
- `evaluate_retriever_configs()`
- `run_full_evaluations()`

These are designed to be composed into a custom evaluation script or notebook workflow.

## Notes

This project is a strong foundation for experimenting with:

- chunking strategies
- retrieval depth (`k` values)
- semantic recall and reranking
- answerability detection
- model selection for policy QA systems

## Contribution

Contributions are welcome. Potential improvements include:

- adding more policy documents
- benchmarking additional retrieval strategies
- improving evaluation metrics and labeling
- adding better infrastructure for automated comparisons
- packaging the project as a reusable CLI or notebook-driven tool

If you want to extend the project, the most natural starting points are the evaluation questions, retriever configs, and orchestration modules.