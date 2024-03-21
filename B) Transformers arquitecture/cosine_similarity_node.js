
require('dotenv').config();
const { OpenAI } = require('openai');

const openai = new OpenAI(process.env.OPENAI_API_KEY);

async function getEmbeddings(words) {
  const embeddings = await Promise.all(
    words.map(async (word) => {
      const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: word,
        encoding_format: "float",
      });
      return response.data[0].embedding;
    })
  );
  return embeddings;
}

async function cosineSimilarity(vectorA, vectorB) {
  // Calculate dot product of the two vectors
  const dotProduct = vectorA.reduce((acc, value, index) => acc + value * vectorB[index], 0);

  // Calculate magnitudes of the vectors
  const magnitudeA = Math.sqrt(vectorA.reduce((acc, value) => acc + value ** 2, 0));
  const magnitudeB = Math.sqrt(vectorB.reduce((acc, value) => acc + value ** 2, 0));

  // Calculate cosine similarity
  const similarity = dotProduct / (magnitudeA * magnitudeB);

  return similarity;
}

async function main() {
  try {
    const words = ["king", "queen"];

    // Get embeddings for the words
    const embeddings = await getEmbeddings(words);

    // Calculate cosine similarity
    const similarity = await cosineSimilarity(embeddings[0], embeddings[1]);

    console.log(`Cosine similarity between "king" and "queen": ${similarity}`);
  } catch (error) {
    console.error("Error:", error.message);
  }
}

main();
