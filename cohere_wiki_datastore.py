# IMPORT NECESSARY LIBRARIES FOR HANDLING DATASETS, FILE OPERATIONS, AND EMBEDDINGS
from datasets import load_dataset
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import shutil
from langchain_community.embeddings import CohereEmbeddings

# DEFINE THE PATH WHERE THE CHROMA DATA WILL BE STORED
CHROMA_PATH = "./chroma"


# FUNCTION TO SAVE DOCUMENTS INTO CHROMA FOR PERSISTENT STORAGE
def save_to_chroma(documents):
    # IF THE CHROMA PATH EXISTS, DELETE IT TO PREVENT DUPLICATIONS
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # INITIALIZE THE PERSISTENT CLIENT AND CREATE OR GET THE COLLECTION
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection("wiki")

    # SET THE EMBEDDING FUNCTION USING COHERE EMBEDDINGS
    embedding_function = CohereEmbeddings(model='multilingual-22-12')

    # ADD DOCUMENTS TO THE COLLECTION
    for document in documents:
        collection.add(embeddings=[document['emb']], documents=[document['text']], ids=[str(document['id'])])

    # INITIALIZE CHROMA WITH THE PERSISTENT CLIENT, COLLECTION NAME, EMBEDDING FUNCTION, AND STORAGE DIRECTORY
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="collection_name",
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH
    )

    # PERSIST THE CHROMA DATA TO DISK
    langchain_chroma.persist()

    # PRINT CONFIRMATION MESSAGE
    print("DONE SAVING")


# FUNCTION TO LOAD DOCUMENTS FROM A DATASET
def load_documents():
    # DEFINE THE MAXIMUM NUMBER OF DOCUMENTS TO LOAD
    max_docs = 1000

    # STREAM DOCUMENTS FROM A SPECIFIED DATASET
    docs_stream = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)
    docs = []

    # COLLECT DOCUMENTS UP TO THE MAXIMUM NUMBER
    for doc in docs_stream:
        docs.append(doc)
        if len(docs) >= max_docs:
            break
    return docs


# FUNCTION TO GENERATE A DATA STORE BY LOADING AND SAVING DOCUMENTS
def generate_data_store():
    documents = load_documents()
    save_to_chroma(documents)


# MAIN EXECUTION BLOCK TO RUN THE DATA STORE GENERATION
if __name__ == "__main__":
    generate_data_store()
