import os
import math
#import numpy as np
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Get the embedding from OpenAI model
response = client.embeddings.create(
    input="black",
    model="text-embedding-3-small"
)

#y = np.array(response.data[0].embedding  )
y= response.data[0].embedding
#print(type(y))
#print(y)

# Assuming you have pos and d_model values
pos = 1
d_model = 512

pe = [[0 for _ in range(512)]]
pc = [[0 for _ in range(512)]]

for i in range(0, 512, 2):
    pe[0][i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
    pc[0][i] = (y[i] * math.sqrt(d_model)) + pe[0][i]
    pe[0][i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
    pc[0][i + 1] = (y[i + 1] * math.sqrt(d_model)) + pe[0][i + 1]

print(pc)