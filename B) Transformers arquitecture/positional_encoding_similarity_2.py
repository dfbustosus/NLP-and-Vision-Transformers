import math
import numpy as np

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def positional_encoding(pos, pe, d_model=512):
    for i in range(0, d_model, 2):
        pe[i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
        pe[i+1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe

# Define d_model
d_model = 512

# Initialize positional encoding arrays
pos_2_output = np.zeros(d_model)
pos_10_output = np.zeros(d_model)

# Calculate positional encoding for pos = 2
positional_encoding(2, pos_2_output)

# Calculate positional encoding for pos = 10
positional_encoding(10, pos_10_output)

# Calculate cosine similarity
similarity_score = cosine_similarity(pos_2_output, pos_10_output)
print(f"Cosine similarity between pos (2) and pos (10): {similarity_score}")
