# IMPORT NECESSARY MODULES AND CLASSES FROM LANGCHAIN AND OTHER PACKAGES
from langchain_community.document_loaders import PyPDFLoader  # USED TO LOAD PDF DOCUMENTS
from langchain_community.document_loaders import DirectoryLoader  # USED TO LOAD DOCUMENTS FROM A DIRECTORY
from langchain.text_splitter import RecursiveCharacterTextSplitter  # USED TO SPLIT TEXT INTO CHUNKS
from langchain.schema import Document  # DOCUMENT SCHEMA FROM LANGCHAIN
from langchain.vectorstores.chroma import Chroma  # CHROMA VECTOR STORE FOR DOCUMENT EMBEDDING STORAGE
from langchain_openai import OpenAIEmbeddings  # OPENAI EMBEDDINGS FOR DOCUMENT VECTOR EMBEDDINGS
import os  # USED FOR OPERATING SYSTEM INTERACTIONS LIKE PATH MANAGEMENT
import shutil  # USED FOR FILE OPERATIONS SUCH AS DELETING DIRECTORIES
from langchain_community.embeddings import CohereEmbeddings # COHERE EMBEDDINGS
# SET THE PATHS FOR LOADING DATA AND SAVING THE CHROMA DATABASE
PATH_TO_DATA = "./Data/Papers/"  # DIRECTORY CONTAINING PDF DOCUMENTS TO LOAD
CHROMA_PATH = "./chroma"  # PATH WHERE THE CHROMA DATABASE WILL BE STORED

# FUNCTION TO LOAD DOCUMENTS FROM A DIRECTORY
def load_documents():
    loader = DirectoryLoader(PATH_TO_DATA, glob="*.pdf")  # INITIALIZES A LOADER FOR PDF FILES IN THE SPECIFIED PATH
    documents = loader.load()  # LOADS THE DOCUMENTS AND RETURNS THEM
    return documents

# FUNCTION TO SPLIT LOADED DOCUMENTS INTO MANAGEABLE TEXT CHUNKS
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # SIZE OF EACH TEXT CHUNK
        chunk_overlap=100,  # NUMBER OF CHARACTERS TO OVERLAP BETWEEN CHUNKS
        length_function=len,  # FUNCTION USED TO MEASURE CHUNK LENGTH
        add_start_index=True,  # INCLUDES THE START INDEX OF EACH CHUNK IN THE DOCUMENT
    )
    chunks = text_splitter.split_documents(documents)  # SPLITS DOCUMENTS INTO CHUNKS BASED ON THE SPECIFIED PARAMETERS
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")  # PRINTS THE NUMBER OF DOCUMENTS AND CHUNKS CREATED

    document = chunks[10]  # EXAMPLE TO ACCESS THE 11TH CHUNK
    print(document.page_content)  # PRINTS THE CONTENT OF THE CHUNK
    print(document.metadata)  # PRINTS THE METADATA ASSOCIATED WITH THE CHUNK

    return chunks  # RETURNS THE LIST OF TEXT CHUNKS

# FUNCTION TO SAVE THE TEXT CHUNKS INTO A CHROMA VECTOR DATABASE
def save_to_chroma(chunks):
    # CLEAR OUT THE CHROMA DATABASE DIRECTORY IF IT EXISTS TO START FRESH
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # CREATE A NEW CHROMA DATABASE FROM THE DOCUMENT CHUNKS WITH OPENAI EMBEDDINGS AND PERSIST IT TO DISK
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()  # PERSISTS THE DATABASE TO THE SPECIFIED DIRECTORY
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")  # PRINTS THE NUMBER OF CHUNKS SAVED

# FUNCTION TO ORCHESTRATE THE DATA LOADING, TEXT SPLITTING, AND SAVING PROCESS
def generate_data_store():
    documents = load_documents()  # LOADS DOCUMENTS FROM THE SPECIFIED PATH
    chunks = split_text(documents)  # SPLITS THE LOADED DOCUMENTS INTO TEXT CHUNKS
    print(chunks[0])  # PRINTS THE FIRST CHUNK FOR INSPECTION
    save_to_chroma(chunks)  # SAVES THE CHUNKS INTO A CHROMA DATABASE

# ENTRY POINT OF THE SCRIPT
if __name__ == "__main__":
    generate_data_store()  # CALLS THE FUNCTION TO START THE PROCESS
