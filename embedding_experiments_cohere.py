from langchain_community.embeddings import CohereEmbeddings


if __name__ == '__main__':
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    vector = embeddings.embed_query("car")
    print(vector)
    print(len(vector))
