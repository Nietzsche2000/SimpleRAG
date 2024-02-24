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
    num = 0
    print("BEGIN MAPPING")
    doc_embds = list(map(lambda x: x['emb'], documents))
    doc_texts = list(map(lambda x: x['text'], documents))
    doc_ids = list(map(lambda x: str(x['id']), documents))
    print("FINISH MAPPING")
    # for document in documents:
    #     print("ADDING DOCS: " + str(num))
    #     doc_embds = list(map(lambda x: x['emb'], documents))
    # collection.add(embeddings=[document['emb']], documents=[document['text']], ids=[str(document['id'])])
    for i in range(0, len(doc_embds) - 40000, 40000):
        print(i)
        j = i + 40000
        collection.add(embeddings=doc_embds[i:j], documents=doc_texts[i:j], ids=doc_ids[i:j])
        # num += 1
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
    max_docs = 40000

    # STREAM DOCUMENTS FROM A SPECIFIED DATASET
    print("INIT DOWNLOAD DATASET")
    docs_stream = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=False)
    docs = []
    print("FINISHED DOWNLOAD DATASET")
    # COLLECT DOCUMENTS UP TO THE MAXIMUM NUMBER
    # for doc in docs_stream:
    #     docs.append(doc)
    #     if len(docs) >= max_docs:
    #         break
    return docs_stream


# FUNCTION TO GENERATE A DATA STORE BY LOADING AND SAVING DOCUMENTS
def generate_data_store():
    documents = load_documents()
    save_to_chroma(documents)


# MAIN EXECUTION BLOCK TO RUN THE DATA STORE GENERATION
if __name__ == "__main__":
    generate_data_store()


# (base) monishwaran@DESKTOP-VNSR1CN:/mnt/c/Users/Monishwaran/Desktop/RAG_System_Cohere/SimpleRAG$
# (base) monishwaran@DESKTOP-VNSR1CN:/mnt/c/Users/Monishwaran/Desktop/RAG_System_Cohere/SimpleRAG$ ^C
# (base) monishwaran@DESKTOP-VNSR1CN:/mnt/c/Users/Monishwaran/Desktop/RAG_System_Cohere/SimpleRAG$ python cohere_wiki_datastore.py
# /home/monishwaran/anaconda3/lib/python3.8/site-packages/scipy/__init__.py:138: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5)
#   warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion} is required for this version of "
# Traceback (most recent call last):
#   File "cohere_wiki_datastore.py", line 82, in <module>
#     generate_data_store()
#   File "cohere_wiki_datastore.py", line 77, in generate_data_store
#     save_to_chroma(documents)
#   File "cohere_wiki_datastore.py", line 40, in save_to_chroma
#     collection.add(embeddings=doc_embds, documents=doc_texts, ids=doc_ids)
#   File "/home/monishwaran/anaconda3/lib/python3.8/site-packages/chromadb/api/models/Collection.py", line 168, in add
#     self._client._add(ids, self.id, embeddings, metadatas, documents, uris)
#   File "/home/monishwaran/anaconda3/lib/python3.8/site-packages/chromadb/telemetry/opentelemetry/__init__.py", line 127, in wrapper
#     return f(*args, **kwargs)
#   File "/home/monishwaran/anaconda3/lib/python3.8/site-packages/chromadb/api/segment.py", line 361, in _add
#     validate_batch(
#   File "/home/monishwaran/anaconda3/lib/python3.8/site-packages/chromadb/api/types.py", line 488, in validate_batch
#     raise ValueError(
# ValueError: Batch size 485859 exceeds maximum batch size 41666
