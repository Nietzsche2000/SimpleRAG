# IMPORT NECESSARY MODULES AND CLASSES FOR COMMAND LINE INTERFACE, DATA HANDLING, AND LANGCHAIN INTEGRATIONS
import argparse  # PARSE COMMAND LINE ARGUMENTS
from dataclasses import dataclass  # CREATE DATA CLASSES
from langchain.vectorstores.chroma import Chroma  # STORE DOCUMENT EMBEDDINGS
from langchain_openai import OpenAIEmbeddings  # VECTOR EMBEDDINGS FROM OPENAI
from langchain_openai import ChatOpenAI  # GENERATE RESPONSES USING OPENAI CHAT MODEL
from langchain.prompts import ChatPromptTemplate  # CREATE CHAT PROMPTS
from langchain_community.embeddings import CohereEmbeddings  # VECTOR EMBEDDINGS FROM COHERE
import chromadb  # CHROMA DATABASE OPERATIONS

# STORE PATH FOR THE CHROMA DATABASE
CHROMA_PATH = "./chroma"

# FORMAT CHAT CONTEXT AND QUESTION
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# EXECUTE SCRIPT
def main():
    # INITIALIZE COMMAND LINE INTERFACE WITH QUERY TEXT ARGUMENT
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="QUERY TEXT")  # QUERY TEXT ARGUMENT
    args = parser.parse_args()  # PARSE COMMAND LINE ARGUMENTS
    query_text = args.query_text  # EXTRACT QUERY TEXT

    print("QUERY TEXT: " + query_text)  # DISPLAY QUERY TEXT

    # SET UP DATABASE FOR SIMILARITY SEARCH
    embedding_function = CohereEmbeddings(model='multilingual-22-12')  # EMBEDDING FUNCTION
    persistent_client = chromadb.PersistentClient()
    db = Chroma(
        client=persistent_client,
        collection_name="wiki",
        embedding_function=embedding_function,
    )
    results = db.similarity_search_with_relevance_scores(query_text, k=3)  # SIMILAR DOCUMENTS SEARCH
    print("RESULTS: " + str(results))  # DISPLAY SEARCH RESULTS

    # COMBINE CONTENT OF TOP DOCUMENTS FOR CONTEXT
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)  # PROMPT TEMPLATE
    prompt = prompt_template.format(context=context_text, question=query_text)  # FORMAT PROMPT

    print("PROMPT: " + prompt)  # DISPLAY FORMATTED PROMPT

    model = ChatOpenAI()  # CHAT MODEL
    response_text = model.predict(prompt)  # GENERATE RESPONSE

    # FORMAT AND DISPLAY RESPONSE AND SOURCES
    sources = [doc.metadata.get("source", None) for doc, _score in results]  # EXTRACT SOURCES
    formatted_response = f"Response: {response_text}\nSources: {sources}"  # FINAL RESPONSE AND SOURCES
    print(formatted_response)  # DISPLAY FORMATTED RESPONSE

# SCRIPT ENTRY POINT
if __name__ == "__main__":
    main()  # RUN MAIN FUNCTION
