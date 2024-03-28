import os
import math
import numpy as np
from openai import OpenAI

# Function to calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Get the embeddings for both words
words = ["black", "brown"]
embeddings = []

for word in words:
    response = client.embeddings.create(
        input=word,
        model="text-embedding-3-small"
    )
    embeddings.append(response.data[0].embedding)

# Assuming you have pos and d_model values
pos = 1
d_model = 512

# Calculate the pc values for both words
pcs = []

for embedding in embeddings:
    pe = np.zeros((1, 512))
    pc = np.zeros((1, 512))
    for i in range(0, 512, 2):
        pe[0, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
        pc[0, i] = (embedding[i] * math.sqrt(d_model)) + pe[0, i]
        pe[0, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
        pc[0, i + 1] = (embedding[i + 1] * math.sqrt(d_model)) + pe[0, i + 1]
    pcs.append(pc)

# Calculate cosine similarity between pc values of "black" and "brown"
similarity = cosine_similarity(pcs[0].flatten(), pcs[1].flatten())
print("Cosine similarity between pc(black) and pc(brown):", similarity)
