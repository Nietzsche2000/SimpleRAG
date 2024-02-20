# IMPORT NECESSARY MODULES AND CLASSES FOR COMMAND LINE INTERFACE, DATA HANDLING, AND LANGCHAIN INTEGRATIONS
import argparse  # USED FOR PARSING COMMAND LINE ARGUMENTS
from dataclasses import dataclass  # USED FOR CREATING DATA CLASSES
from langchain.vectorstores.chroma import Chroma  # CHROMA VECTOR STORE FOR DOCUMENT EMBEDDING STORAGE
from langchain_openai import OpenAIEmbeddings  # OPENAI EMBEDDINGS FOR DOCUMENT VECTOR EMBEDDINGS
from langchain_openai import ChatOpenAI  # OPENAI CHAT MODEL FOR GENERATING RESPONSES
from langchain.prompts import ChatPromptTemplate  # TEMPLATE FOR CREATING CHAT PROMPTS

# DEFINE PATH WHERE THE CHROMA DATABASE IS STORED
CHROMA_PATH = "chroma"

# DEFINE A PROMPT TEMPLATE FOR FORMATTING THE CHAT CONTEXT AND QUESTION
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# MAIN FUNCTION TO EXECUTE THE SCRIPT
def main():
    # INITIALIZE COMMAND LINE INTERFACE WITH ARGUMENT FOR QUERY TEXT
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")  # DEFINE QUERY TEXT ARGUMENT
    args = parser.parse_args()  # PARSE ARGUMENTS FROM COMMAND LINE
    query_text = args.query_text  # EXTRACT QUERY TEXT FROM PARSED ARGUMENTS

    print("QUERY TEXT: " + query_text)  # PRINT THE QUERY TEXT

    # PREPARE THE DATABASE FOR SIMILARITY SEARCH
    embedding_function = OpenAIEmbeddings()  # INITIALIZE EMBEDDING FUNCTION
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)  # INITIALIZE CHROMA DATABASE

    # PERFORM SIMILARITY SEARCH IN THE DATABASE BASED ON QUERY TEXT
    results = db.similarity_search_with_relevance_scores(query_text, k=3)  # SEARCH FOR SIMILAR DOCUMENTS

    print("RESULTS: " + str(results))  # PRINT THE SEARCH RESULTS

    # FORMAT THE CONTEXT TEXT BY COMBINING THE CONTENT OF TOP DOCUMENTS
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)  # INITIALIZE PROMPT TEMPLATE
    prompt = prompt_template.format(context=context_text,
                                    question=query_text)  # FORMAT PROMPT WITH CONTEXT AND QUESTION

    print("PROMPT: " + prompt)  # PRINT THE FORMATTED PROMPT

    model = ChatOpenAI()  # INITIALIZE THE CHAT MODEL
    response_text = model.predict(prompt)  # GENERATE RESPONSE BASED ON THE PROMPT

    # FORMAT SOURCES AND RESPONSE FOR PRINTING
    sources = [doc.metadata.get("source", None) for doc, _score in results]  # EXTRACT SOURCES FROM RESULTS
    formatted_response = f"Response: {response_text}\nSources: {sources}"  # FORMAT THE FINAL RESPONSE AND SOURCES
    print(formatted_response)  # PRINT THE FORMATTED RESPONSE


# ENTRY POINT OF THE SCRIPT
if __name__ == "__main__":
    main()  # EXECUTE MAIN FUNCTION
