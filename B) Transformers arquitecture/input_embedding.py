import os
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.embeddings.create(
    input="The black cat sat on the couch and the brown dog slept on the couch",
    model="text-embedding-3-small"
)

print(response.data[0].embedding)