
import tensorflow as tf
import numpy as np
import time

# define the sequence length and representation dimensionality
n =  32768
d = 12288

# define the inputs
input_seq = tf.random.normal((n, d), dtype=tf.float32)

# simulation of self-attention layer O(n^2*d)
start_time = time.time()
_ = tf.matmul(input_seq, input_seq, transpose_b=True)

at = time.time() - start_time
print(f"Self-attention computation time: {at} seconds")

# Output expected
# Self-attention computation time: 27.117244005203247 seconds
