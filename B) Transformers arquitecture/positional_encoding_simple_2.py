import math
import numpy as np

def positional_encoding(pos, pe, d_model=512):
    for i in range(0, d_model, 2):
        pe[0][i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
        pe[0][i+1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe

# Define d_model
d_model = 512

# Initialize positional encoding array
pe = np.zeros((1, d_model))

# Calculate positional encoding for pos = 2
pos_2_output = positional_encoding(2, pe.copy())

# Calculate positional encoding for pos = 10
pos_10_output = positional_encoding(10, pe.copy())

# Print the outputs
print("Positional encoding for pos = 2:")
print(pos_2_output)

print("\nPositional encoding for pos = 10:")
print(pos_10_output)
