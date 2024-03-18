import tensorflow as tf
import numpy as np
import time

# define the sequence length and representation dimensionality
n = 512
d = 512

# define the inputs
input_seq = tf.random.normal((n, d), dtype=tf.float32)

# simulation of self-attention layer O(n^2*d)
start_time = time.time()
_ = tf.matmul(input_seq, input_seq, transpose_b=True)
at = time.time() - start_time
print(f"Self-attention computation time: {at} seconds")

# simulation of recurrent layer O(n*d^2)
start_time = time.time()
hidden_state = np.zeros((n, d), dtype=np.float32)
for i in range(n):
    for j in range(d):
        for k in range(d):
            hidden_state[i, j] += input_seq[i, j].numpy() * hidden_state[min(i,k), j]
            ct = time.time() - start_time
            if ct>at*10:
              break

rt = time.time() - start_time
print(f"Recurrent layer computation time: {rt} seconds")

# Calculate the total
total = at + rt

# Calculate the percentage of at
percentage_at = round((at / total) * 100, 2)

# Output the result
print(f"The percentage of self-attention computation in the sum of self-attention and recurrent computation is {percentage_at}%")

# Output expected
#Self-attention computation time: 0.10626077651977539 seconds
#Recurrent layer computation time: 66.53181290626526 seconds
#The percentage of self-attention computation in the sum of self-attention and recurrent computation is 0.16%