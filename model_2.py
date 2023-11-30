import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_sentence_embedding(sentence):

    embedding = embed([sentence])[0].numpy()
    return embedding

def calculate_similarity(embedding1, embedding2):

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def main():
    sentence1 = input("\n\n\n첫 번째 문장을 입력하세요: ")
    sentence2 = input("두 번째 문장을 입력하세요: ")


    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)


    similarity = calculate_similarity(embedding1, embedding2)


    print(f"\n첫 번째 문장의 의미 벡터: {embedding1}")
    print(f"두 번째 문장의 의미 벡터: {embedding2}")
    print(f"\n\n의미적 유사도: {similarity:.5f}")
    

if __name__ == "__main__":
    main()