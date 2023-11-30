import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

file_name = './sample.xlsx'
df = pd.read_excel(file_name, sheet_name='samples')

sample_sentence1 = str(df['Text1'][0])
sample_sentence2 = str(df['Text2'][0])

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_sentence_embedding(sentence):

    embedding = embed([sentence])[0].numpy()
    return embedding

def calculate_similarity(embedding1, embedding2):

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def main():
    print(f'\n\nsample_sentence1은 "{sample_sentence1}"입니다.')
    print(f'\nsample_sentence2은 "{sample_sentence2}"입니다.')
    

    embedding1 = get_sentence_embedding(sample_sentence1)
    embedding2 = get_sentence_embedding(sample_sentence2)


    similarity = calculate_similarity(embedding1, embedding2)


    print(f"\nsample_sentence1 embedding: {embedding1}")
    print(f"sample_sentence2 embedding: {embedding2}")
    print(f"\n\nSemantic similarity: {similarity:.5f}")
    

if __name__ == "__main__":
    main()