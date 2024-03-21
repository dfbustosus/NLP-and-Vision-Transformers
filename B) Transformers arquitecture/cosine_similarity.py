import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client with your API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Function to get embeddings of words
def get_embeddings(words):
    embeddings = []
    for word in words:
        response = client.embeddings.create(
            input=word,
            model="text-embedding-3-small",
            encoding_format="float"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# Function to calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def main():
    words = ["king", "queen"]

    # Get embeddings for the words
    embeddings = get_embeddings(words)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(embeddings[0], embeddings[1])

    print(f"Cosine similarity between 'king' and 'queen': {similarity_score}")

if __name__ == "__main__":
    main()
