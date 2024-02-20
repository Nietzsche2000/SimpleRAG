from langchain_openai import OpenAIEmbeddings


if __name__ == '__main__':
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("Car")
    print(vector)
    print(len(vector))
