# SimpleRAG
A simple Retrieval Augmented Generation (RAG) pipeline for LLMs. It consists of two main components: a document datastore (`datastore.py`) for managing and storing document embeddings, and a content generator (`generator.py`) that utilizes a vector database to generate responses based on user queries.

## Features

- **Document Datastore (`datastore.py`)**: Load documents, split them into manageable chunks, and store them in a Chroma vector database using LangChain and OpenAI embeddings.
- **Content Generator (`generator.py`)**: Query the Chroma datastore and generate relevant content based on the documents found, utilizing OpenAI's GPT model for generating responses.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Required Python packages: `langchain`, `langchain_openai`, `argparse`, `dataclasses`

You can install the necessary packages using pip:

```bash
pip install langchain langchain_openai argparse dataclasses
```

### Installation

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-name>
```

### Usage

#### Setting Up the Document Datastore

1. Place your PDF documents in the `./Data/Papers/` directory.
2. Run `datastore.py` to process and store your documents:

```bash
python datastore.py
```

#### Generating Content

Use `generator.py` to query the datastore and generate responses:

```bash
python generator.py "<your query here>"
```

Replace `<your query here>` with the query you want to generate content for.

## Configuration

- **Data Path**: The default path for loading documents is `./Data/Papers/`. You can change this path in `datastore.py`.
- **Chroma Database Path**: The default path for the Chroma vector database is `./chroma`. This can be modified in both scripts as needed.
