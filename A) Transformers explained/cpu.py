#Computational times of complexity per layer
# Comparing the computational time between:
# self attention = O(n^2 * d)
#and
# recurrent = O(n * d^2)
import numpy as np
import time

# define the sequence length and representation dimensionality
n = 512 # number of words in a sequence
d = 512 # number of dimensions (features represenitng a word)
# Special case n= d which means O(n^2 *d) = O(n* d^2)= 512*512*512= 134217728 operations
# Therefore both attention and recurrent layers have the same number of operations to perform the computational complexity time

# define the inputs
input_seq = np.random.rand(n, d)

# simulation of self-attention layer O(n^2*d) with a matrix multiplication with start time
start_time = time.time()
for i in range(n):
    for j in range(n):
        _ = np.dot(input_seq[i], input_seq[j])
at=time.time()-start_time
print(f"Self-attention computation time: {time.time() - start_time} seconds")

# simulation of recurrent layer O(n*d^2)
start_time = time.time()
hidden_state = np.zeros(d)
for i in range(n):
    for j in range(d):
        for k in range(d):
            hidden_state[j] += input_seq[i, j] * hidden_state[k]
rt=time.time()-start_time
print(f"Recurrent layer computation time: {time.time() - start_time} seconds")

# Calculate the total
total = at + rt

# Calculate the percentage of at
percentage_at = round((at / total) * 100,2)

# Output the result
print(f"The percentage of 'computational time for attention' in the sum of 'attention' and 'recurrent' is {percentage_at}%")

# Calculate x, which is the ratio of rt to at
x = round(rt / at,2)

# Output
# Self-attention computation time: 0.6672823429107666 seconds
# Recurrent layer computation time: 112.06018710136414 seconds
# The percentage of 'computational time for attention' in the sum of 'attention' and 'recurrent' is 0.59%