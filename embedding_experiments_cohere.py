from langchain_community.embeddings import CohereEmbeddings


if __name__ == '__main__':
    embeddings = CohereEmbeddings(model='multilingual-22-12')
    vector = embeddings.embed_query("car")
    print(vector)
    print(len(vector))
